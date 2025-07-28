import collections
import torch
import torch.nn.functional as F
import numpy as np
import torch
from dllm_cache import FeatureCache
from slow_fast_sampling.semantic_weight_attention import SemanticWeightSelfAttention

def extract_token_features(x, logits, attention_weights, prompt_length, mask_id, prompt_mask):
    """
    x: [batch, seq_len] 当前 token 序列
    logits: [batch, seq_len, vocab_size] 当前 logits
    attention_weights: [batch, nhead, seq_len, seq_len] 当前注意力权重
    prompt_length: int
    mask_id: int
    prompt_mask: [batch, seq_len] 1表示prompt区域，0表示生成区域
    返回: [num_mask, feature_dim]，每行为一个 [MASK] token 的特征
    """
    batch, seq_len = x.shape
    nhead = attention_weights.shape[1]
    features = []
    for b in range(batch):
        for i in range(prompt_length, seq_len):
            if x[b, i] == mask_id:
                # softmax 置信度
                prob = torch.softmax(logits[b, i], dim=-1)
                conf = prob.max().item()
                # 注意力熵
                entropy = 0.0
                for h in range(nhead):
                    attn = attention_weights[b, h, i, :]
                    attn = attn / (attn.sum() + 1e-8)
                    entropy += -(attn * (attn + 1e-8).log()).sum().item()
                entropy = entropy / nhead
                # prompt mass
                prompt_mass = 0.0
                for h in range(nhead):
                    attn = attention_weights[b, h, i, :prompt_length]
                    prompt_mass += attn.sum().item()
                prompt_mass = prompt_mass / nhead
                # 位置归一化
                pos = i / seq_len
                # mask_type
                mask_type = 0 if i < prompt_length else 1
                features.append([conf, 1-entropy, prompt_mass, pos, mask_type])
    return torch.tensor(features, dtype=torch.float32)

class SlowFastSampler:
    def __init__(
        self,
        model,
        gen_kwargs,
        mask_id=126336,
        temperature=0.0,
        cfg_scale=0.0,
        semantic_weight_module=None,  # 新增
    ):
        self.model = model
        self.mask_id = mask_id
        self.temperature = temperature
        self.cfg_scale = cfg_scale
        self.k_exploration_steps = gen_kwargs.get("k_exploration_steps", 6)
        self.cycle_len_confidence_threshold = gen_kwargs.get("cycle_len_confidence_threshold", 0.3)
        self.cycle_length_stability_window = gen_kwargs.get("cycle_length_stability_window", 2)
        self.cycle_length_stability_std_dev_threshold = gen_kwargs.get("cycle_length_stability_std_dev_threshold", 1.0)
        self.high_confidence_threshold = gen_kwargs.get("high_confidence_threshold", 0.9)
        self.num_important_low_confidence_tokens = gen_kwargs.get("num_important_low_confidence_tokens", 3)
        self.max_sub_cycles_per_block = gen_kwargs.get("max_sub_cycles_per_block", 256)
        self.gen_length=gen_kwargs.get("gen_length", 128)
        self.block_length=gen_kwargs.get("block_length", 128)
        self.semantic_weight_module = semantic_weight_module  # 新增
    
    def add_gumbel_noise(self,logits):
        """
        The Gumbel max is a method for sampling categorical distributions.
        According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
        Thus, we use float64.
        """
        if self.temperature == 0:
            return logits.exp()
        noise = torch.rand_like(logits)
        gumbel_noise = (-torch.log(noise)) ** self.temperature
        return logits.exp() / gumbel_noise
    def get_num_tokens_for_phase1_step(self,current_sub_cycle_mask):
        batch_size = current_sub_cycle_mask.shape[0]
        return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device) # try to fill 1 tokens

    def get_num_tokens_for_phase3_step(self,current_sub_cycle_mask):
        batch_size = current_sub_cycle_mask.shape[0]
        return torch.full((batch_size,), 1, dtype=torch.long, device=current_sub_cycle_mask.device) # try to fill 1 tokens

    def slow_phase(self, x, prompt_length, block_idx, last_sub_cycle_length_per_item, actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords, prompt_index_full_x, attention_mask):
        batch_size = x.shape[0]
        block_start_in_gen = block_idx * self.block_length
        block_end_in_gen = (block_idx + 1) * self.block_length
        sub_cycle_determined_per_item = torch.zeros(batch_size, dtype=torch.bool, device=x.device)
        history_per_item = [collections.deque(maxlen=self.cycle_length_stability_window) for _ in range(batch_size)]

        for k_step in range(self.k_exploration_steps):
            if self.cfg_scale > 0.0: 
                cfg_x = x.clone()
                cfg_x[prompt_index_full_x] = self.mask_id
                logits_main_out = self.model(x, attention_mask=attention_mask, output_attentions=True)
                logits_main = logits_main_out.logits
                attn_main = logits_main_out.attentions[-1] if hasattr(logits_main_out, 'attentions') else None
                cfg_logits_main_out = self.model(cfg_x, attention_mask=attention_mask, output_attentions=True)
                cfg_logits_main = cfg_logits_main_out.logits
                logits_full = logits_main + self.cfg_scale * (logits_main - cfg_logits_main)
                attn = attn_main
            else:
                logits_out = self.model(x, attention_mask=attention_mask, output_attentions=True)
                logits_full = logits_out.logits
                attn = logits_out.attentions[-1] if hasattr(logits_out, 'attentions') else None
            logits_gen_part = logits_full[:, prompt_length:]
            x0_gen = torch.argmax(self.add_gumbel_noise(logits_gen_part), dim=-1)
            p_gen = F.softmax(logits_gen_part, dim=-1)
            x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)

            # 动态置信度融合
            if self.semantic_weight_module is not None and attn is not None:
                # attn: [batch, nhead, seq_len, seq_len]
                prompt_mask = (x[:, :].clone() != self.mask_id).long()
                features = extract_token_features(x, logits_full, attn, prompt_length, self.mask_id, prompt_mask)
                weights = self.semantic_weight_module(features)  # [num_mask, 3]
                # features: [conf, 1-entropy, prompt_mass, pos, mask_type]
                conf = features[:, 0]
                entropy = 1 - features[:, 1]
                pmass = features[:, 2]
                alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]
                semantic_conf = alpha * conf + beta * (1 - entropy) + gamma * pmass
                # 用 semantic_conf 替换 confidence_gen_wide
                # 这里简化处理：将 semantic_conf 按 mask 顺序填入 confidence_gen_wide 的 mask 位置
                confidence_gen_wide = torch.full_like(x0_p_gen, -float('inf'))
                mask_indices = (x[:, prompt_length:] == self.mask_id)
                idx = 0
                for b in range(x.shape[0]):
                    for i in range(x0_p_gen.shape[1]):
                        if mask_indices[b, i]:
                            confidence_gen_wide[b, i] = semantic_conf[idx]
                            idx += 1
            else:
                current_global_mask_index_gen_part = (x[:, prompt_length:] == self.mask_id) 
                confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen, torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))

            #Estimate sub-cycle length (focus on current block)
            for b_idx in range(batch_size):
                if not sub_cycle_determined_per_item[b_idx]:
                    previous_len_item = last_sub_cycle_length_per_item[b_idx].item()
                    observation_abs_start_in_gen = block_start_in_gen + previous_len_item
                    observation_abs_end_in_gen = block_end_in_gen 
                    increment_len = 0

                    if observation_abs_start_in_gen < observation_abs_end_in_gen:
                        confidence_in_observation_scope = confidence_gen_wide[b_idx, observation_abs_start_in_gen : observation_abs_end_in_gen]
                        if confidence_in_observation_scope.numel() > 0: 
                            above_thresh_indices_in_scope = (confidence_in_observation_scope >= self.cycle_len_confidence_threshold).nonzero(as_tuple=True)[0] 
                            if len(above_thresh_indices_in_scope) > 0:
                                farthest_idx_in_scope = above_thresh_indices_in_scope.max().item()
                                increment_len = farthest_idx_in_scope + 1
                            else:
                                increment_len = 1 
                        else:
                            pass 
                    else:
                        increment_len = 0

                    est_len = previous_len_item + increment_len
                    est_len = max(1, est_len) 
                    est_len = min(est_len, self.block_length) 
                    history_per_item[b_idx].append(est_len)

                    if len(history_per_item[b_idx]) >= self.cycle_length_stability_window:
                        hist_np = np.array(list(history_per_item[b_idx]))
                        if np.std(hist_np) < self.cycle_length_stability_std_dev_threshold:
                            det_len = int(history_per_item[b_idx][-1])
                            actual_sub_cycle_length_per_item[b_idx] = max(1, min(det_len, self.block_length))
                            sub_cycle_determined_per_item[b_idx] = True
                        else:
                            det_len = int(np.mean(hist_np))
                            actual_sub_cycle_length_per_item[b_idx] = max(1, min(det_len, self.block_length))
                            sub_cycle_determined_per_item[b_idx] = False if k_step < self.k_exploration_steps - 1 else True                 

            #Initial fill
            num_to_fill_p1 = self.get_num_tokens_for_phase1_step(mask_in_current_block_abs_coords) 
            transfer_mask_p1 = torch.zeros_like(x0_gen, dtype=torch.bool)
            for b_idx in range(batch_size):
                previous_len_item_fill = last_sub_cycle_length_per_item[b_idx].item()
                fill_op_abs_start_in_gen = block_start_in_gen + previous_len_item_fill
                fill_op_abs_end_in_gen = block_end_in_gen
                increment_len_p1_fill = 0 
                if fill_op_abs_start_in_gen < fill_op_abs_end_in_gen:
                    conf_in_fill_op_scope = confidence_gen_wide[b_idx, fill_op_abs_start_in_gen : fill_op_abs_end_in_gen]
                    mask_in_fill_op_scope = (x[b_idx, prompt_length + fill_op_abs_start_in_gen : prompt_length + fill_op_abs_end_in_gen] == self.mask_id)
                    if conf_in_fill_op_scope.numel() > 0: 
                        eff_conf_in_fill_op_scope = torch.where(mask_in_fill_op_scope, conf_in_fill_op_scope, torch.tensor(-np.inf, device=x.device, dtype=conf_in_fill_op_scope.dtype))
                        num_masked_in_fill_op_scope = mask_in_fill_op_scope.sum().item()
                        if num_to_fill_p1[b_idx] > 0 and num_masked_in_fill_op_scope > 0:
                            k = min(int(num_to_fill_p1[b_idx].item()), num_masked_in_fill_op_scope)
                            phase1_high_conf_fill_indices = (conf_in_fill_op_scope >= self.high_confidence_threshold) & mask_in_fill_op_scope
                            if phase1_high_conf_fill_indices.any() and phase1_high_conf_fill_indices.sum().item()>1:
                                abs_indices_to_fill = fill_op_abs_start_in_gen + phase1_high_conf_fill_indices.nonzero(as_tuple=True)[0]
                                transfer_mask_p1[b_idx, abs_indices_to_fill] = True
                            else:           
                                if k > 0:
                                    top_k_indices_relative_to_fill_scope = torch.topk(eff_conf_in_fill_op_scope, k=int(k)).indices
                                    abs_indices_to_fill_in_gen = fill_op_abs_start_in_gen + top_k_indices_relative_to_fill_scope
                                    transfer_mask_p1[b_idx, abs_indices_to_fill_in_gen] = True
            x[:, prompt_length:][transfer_mask_p1] = x0_gen[transfer_mask_p1]

        # After k_exploration_steps, if any item's sub-cycle length is not determined, use a fallback.
        for b_idx in range(batch_size):
            if not sub_cycle_determined_per_item[b_idx]:
                if len(history_per_item[b_idx]) > 0: # Use average of what was gathered
                    actual_sub_cycle_length_per_item[b_idx] = max(1, min(int(np.mean(list(history_per_item[b_idx]))), self.block_length))
                else: # Absolute fallback
                    actual_sub_cycle_length_per_item[b_idx] = self.block_length // 2 # Or some other default
                sub_cycle_determined_per_item[b_idx] = True # Mark as determined for next phases
        return x
        
    def fast_phase(self, x, prompt_length, block_idx, last_sub_cycle_length_per_item, actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords, prompt_index_full_x, attention_mask):
        batch_size = x.shape[0]
        phase_2_and_3_calls = 0
        block_start_in_gen = block_idx * self.block_length
        block_end_in_gen = (block_idx + 1) * self.block_length
        # cache list
        cache_out_cycle_logits_list = []
        cache_out_cycle_cfg_logits_list = []
        cache_out_cycle_full_logits_list = []
        # cycle list
        active_region_start_check_list = []
        active_region_end_check_list = []
        while True:
            all_p2_active_regions_filled_for_all_items = True
            for b_idx_check in range(batch_size):
                current_cumulative_len_check = actual_sub_cycle_length_per_item[b_idx_check].item()
                previous_cumulative_len_check = last_sub_cycle_length_per_item[b_idx_check].item()
                active_region_start_check_list.append(block_start_in_gen + previous_cumulative_len_check)
                active_region_end_check_list.append(block_start_in_gen + current_cumulative_len_check)
                if active_region_start_check_list[b_idx_check] < active_region_end_check_list[b_idx_check]:
                    mask_in_ar_check = (x[b_idx_check, prompt_length + active_region_start_check_list[b_idx_check] : prompt_length + active_region_end_check_list[b_idx_check]] == self.mask_id)
                    if mask_in_ar_check.any(): # If any mask exists in this item's active region
                        all_p2_active_regions_filled_for_all_items = False
                        break # No need to check other items, we know P2 still has work
            if all_p2_active_regions_filled_for_all_items:
                break 
            
            phase_2_and_3_calls += 1
            
            # model call
            if self.cfg_scale > 0.0: # Simplified CFG
                if phase_2_and_3_calls == 1:
                    cfg_x = x.clone()
                    cfg_x[prompt_index_full_x] = self.mask_id
                    logits_main_out = self.model(x, attention_mask=attention_mask, output_attentions=True)
                    logits_main = logits_main_out.logits
                    attn_main = logits_main_out.attentions[-1] if hasattr(logits_main_out, 'attentions') else None
                    cfg_logits_main_out = self.model(cfg_x, attention_mask=attention_mask, output_attentions=True)
                    cfg_logits_main = cfg_logits_main_out.logits
                    logits_full = logits_main + self.cfg_scale * (logits_main - cfg_logits_main)
                    attn = attn_main
                else:
                    cfg_x = x.clone()
                    cfg_x[prompt_index_full_x] = self.mask_id
                    logits_main_batch = []
                    cfg_logits_main_batch = []
                    attn = None
                    for b_idx_check in range(batch_size):
                        logits_main_part_out = self.model(x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask, output_attentions=True)
                        logits_main_part = logits_main_part_out.logits
                        attn_main_part = logits_main_part_out.attentions[-1] if hasattr(logits_main_part_out, 'attentions') else None
                        cfg_logits_main_part_out = self.model(cfg_x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask, output_attentions=True)
                        cfg_logits_main_part = cfg_logits_main_part_out.logits
                        logits_main_batch.append(torch.cat([logits_main_part[b_idx_check].unsqueeze(0), cache_out_cycle_logits_list[b_idx_check]], dim=1))
                        cfg_logits_main_batch.append(torch.cat([cfg_logits_main_part[b_idx_check].unsqueeze(0), cache_out_cycle_cfg_logits_list[b_idx_check]],dim=1))
                        attn = attn_main_part  # 只取最后一个 batch 的 attn
                    logits_main = torch.cat(logits_main_batch, dim=0)
                    cfg_logits_main = torch.cat(cfg_logits_main_batch, dim=0)
                    logits_full = logits_main + self.cfg_scale * (logits_main - cfg_logits_main)
            else:
                if phase_2_and_3_calls == 1:
                    logits_out = self.model(x, attention_mask=attention_mask, output_attentions=True)
                    logits_full = logits_out.logits
                    attn = logits_out.attentions[-1] if hasattr(logits_out, 'attentions') else None
                    for b_idx_check in range(batch_size):
                        cache_out_cycle_full_logits_list.append(logits_full[b_idx_check, prompt_length + active_region_end_check_list[b_idx_check]:].unsqueeze(0))
                else:
                    logits_full_batch = []
                    attn = None
                    for b_idx_check in range(batch_size):
                        logits_full_item_out = self.model(x[:, :prompt_length + active_region_end_check_list[b_idx_check]], attention_mask=attention_mask, output_attentions=True)
                        logits_full_item = logits_full_item_out.logits
                        attn_item = logits_full_item_out.attentions[-1] if hasattr(logits_full_item_out, 'attentions') else None
                        logits_full_batch.append(torch.cat([logits_full_item[b_idx_check].unsqueeze(0), cache_out_cycle_full_logits_list[b_idx_check]], dim=1))
                        attn = attn_item  # 只取最后一个 batch 的 attn
                    logits_full = torch.cat(logits_full_batch, dim=0)
            
            logits_gen_part = logits_full[:, prompt_length:]
            x0_gen = torch.argmax(self.add_gumbel_noise(logits_gen_part), dim=-1)
            p_gen = F.softmax(logits_gen_part, dim=-1)
            x0_p_gen = torch.gather(p_gen, dim=-1, index=x0_gen.unsqueeze(-1)).squeeze(-1)
            current_global_mask_index_gen_part = (x[:, prompt_length:] == self.mask_id)
            # 动态置信度融合
            if self.semantic_weight_module is not None and attn is not None:
                prompt_mask = (x[:, :].clone() != self.mask_id).long()
                features = extract_token_features(x, logits_full, attn, prompt_length, self.mask_id, prompt_mask)
                weights = self.semantic_weight_module(features)  # [num_mask, 3]
                conf = features[:, 0]
                entropy = 1 - features[:, 1]
                pmass = features[:, 2]
                alpha, beta, gamma = weights[:, 0], weights[:, 1], weights[:, 2]
                semantic_conf = alpha * conf + beta * (1 - entropy) + gamma * pmass
                confidence_gen_wide = torch.full_like(x0_p_gen, -float('inf'))
                mask_indices = (x[:, prompt_length:] == self.mask_id)
                idx = 0
                for b in range(x.shape[0]):
                    for i in range(x0_p_gen.shape[1]):
                        if mask_indices[b, i]:
                            confidence_gen_wide[b, i] = semantic_conf[idx]
                            idx += 1
            else:
                confidence_gen_wide = torch.where(current_global_mask_index_gen_part, x0_p_gen, torch.tensor(-np.inf, device=x.device, dtype=x0_p_gen.dtype))
            transfer_mask_p2_and_p3 = torch.zeros_like(x0_gen, dtype=torch.bool)
            
            for b_idx in range(batch_size):
                sub_cycle_abs_end_in_gen = block_start_in_gen + actual_sub_cycle_length_per_item[b_idx].item()
                sub_cycle_abs_start_in_gen = block_start_in_gen + last_sub_cycle_length_per_item[b_idx].item() 
                

                conf_in_sub_cycle_scope = confidence_gen_wide[b_idx, sub_cycle_abs_start_in_gen:sub_cycle_abs_end_in_gen]
                mask_in_sub_cycle_scope = (x[b_idx, prompt_length + sub_cycle_abs_start_in_gen : prompt_length + sub_cycle_abs_end_in_gen] == self.mask_id)

                high_conf_fill_indices = (conf_in_sub_cycle_scope >= self.high_confidence_threshold) & mask_in_sub_cycle_scope

                # print(f"high_conf_fill_indices{high_conf_fill_indices}")
                
                if high_conf_fill_indices.any() and high_conf_fill_indices.sum().item()>1:
                    abs_indices_to_fill = sub_cycle_abs_start_in_gen + high_conf_fill_indices.nonzero(as_tuple=True)[0]
                    transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True
                else:
                    n2_num_transfer_tokens = self.get_num_tokens_for_phase3_step(mask_in_current_block_abs_coords)
                    eff_conf_sub_cycle = torch.where(mask_in_sub_cycle_scope, conf_in_sub_cycle_scope, torch.tensor(-np.inf, device=x.device, dtype=conf_in_sub_cycle_scope.dtype))
            
                    top_k_indices_relative_to_sub_cycle = torch.topk(eff_conf_sub_cycle, k=int(n2_num_transfer_tokens[b_idx].item())).indices
                    abs_indices_to_fill = sub_cycle_abs_start_in_gen + top_k_indices_relative_to_sub_cycle
                    transfer_mask_p2_and_p3[b_idx, abs_indices_to_fill] = True
                    
            x[:, prompt_length:][transfer_mask_p2_and_p3] = x0_gen[transfer_mask_p2_and_p3] # Update x
        return x
    def generate(self, input_ids, attention_mask):
        with torch.no_grad():
            batch_size, prompt_length = input_ids.shape
            x = torch.full(
                (batch_size, prompt_length + self.gen_length),
                self.mask_id, dtype=torch.long, device=self.model.device,
            )
            x[:, :prompt_length] = input_ids
            prompt_index_full_x = (x != self.mask_id)

            assert self.gen_length % self.block_length == 0
            num_blocks = self.gen_length // self.block_length

            feature_cache = FeatureCache()
            feature_cache.reset_cache(prompt_length,gen_length=self.gen_length)
            
            for block_idx in range(num_blocks):
                block_abs_start_in_x = prompt_length + block_idx * self.block_length
                block_abs_end_in_x = prompt_length + (block_idx + 1) * self.block_length

                current_sub_cycles_in_block = 0
                actual_sub_cycle_length_per_item = torch.full((batch_size,), self.block_length, dtype=torch.long, device=x.device)
                last_sub_cycle_length_per_item = torch.full((batch_size,), 0, dtype=torch.long, device=x.device)
                
                while True:
                    mask_in_current_block_abs_coords = (x[:, block_abs_start_in_x:block_abs_end_in_x] == self.mask_id)
                    if not mask_in_current_block_abs_coords.any():
                        break
                    if current_sub_cycles_in_block >= self.max_sub_cycles_per_block:
                        break
                    
                    current_sub_cycles_in_block += 1
                   
                    x = self.slow_phase(
                        x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                        actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                        prompt_index_full_x, attention_mask)

                    x = self.fast_phase(
                        x, prompt_length, block_idx, last_sub_cycle_length_per_item,
                        actual_sub_cycle_length_per_item, mask_in_current_block_abs_coords,
                        prompt_index_full_x, attention_mask)

                    last_sub_cycle_length_per_item = actual_sub_cycle_length_per_item.clone()

            return x[:, prompt_length:]