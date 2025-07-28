import torch
import torch.nn as nn
import torch.optim as optim
from slow_fast_sampling.semantic_weight_attention import SemanticWeightSelfAttention

# 假设有特征样本集 features: [N, d_model]，静态最优权重 labels: [N, 3]
# 这里用随机数据模拟，实际可用采样日志或人工构造
N = 1000
feature_dim = 5
features = torch.randn(N, feature_dim)
# 静态最优权重 alpha=0.6, beta=0.2, gamma=0.2
labels = torch.tensor([[0.6, 0.2, 0.2]] * N, dtype=torch.float32)

# 构建调权模块
semantic_weight_module = SemanticWeightSelfAttention(d_model=32, nhead=4, num_layers=1)
# 特征升维到 d_model
feature_proj = nn.Linear(feature_dim, 32)

optimizer = optim.Adam(list(semantic_weight_module.parameters()) + list(feature_proj.parameters()), lr=1e-3)
loss_fn = nn.MSELoss()

batch_size = 64
num_epochs = 10

for epoch in range(num_epochs):
    perm = torch.randperm(N)
    features_shuffled = features[perm]
    labels_shuffled = labels[perm]
    for i in range(0, N, batch_size):
        batch_feat = features_shuffled[i:i+batch_size]
        batch_label = labels_shuffled[i:i+batch_size]
        # 升维
        batch_feat_proj = feature_proj(batch_feat)
        # [seq_len, batch, d_model] 这里每个 batch 视为一个 token
        input_feat = batch_feat_proj.unsqueeze(1)  # [batch, 1, d_model]
        input_feat = input_feat.transpose(0, 1)    # [1, batch, d_model]
        out = semantic_weight_module(input_feat).squeeze(0)  # [batch, 3]
        loss = loss_fn(out, batch_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# 保存权重
semantic_weight_module.save("semantic_weight_module.pt")
feature_proj_path = "semantic_weight_feature_proj.pt"
torch.save(feature_proj.state_dict(), feature_proj_path)
print(f"权重已保存到 semantic_weight_module.pt 和 {feature_proj_path}") 