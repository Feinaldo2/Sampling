import torch
import torch.nn as nn

class SemanticWeightSelfAttention(nn.Module):
    def __init__(self, d_model=32, nhead=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=64, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.proj = nn.Linear(d_model, 3)  # 输出 α, β, γ

    def forward(self, features):
        """
        features: [seq_len, batch, d_model] 或 [seq_len, d_model]（单 batch）
        返回: [seq_len, batch, 3] 或 [seq_len, 3]
        """
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [seq_len, 1, d_model]
        x = self.encoder(features)  # [seq_len, batch, d_model]
        weights = self.proj(x)  # [seq_len, batch, 3]
        weights = torch.softmax(weights, dim=-1)
        if weights.shape[1] == 1:
            weights = weights.squeeze(1)  # [seq_len, 3]
        return weights

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, map_location=None):
        self.load_state_dict(torch.load(path, map_location=map_location)) 