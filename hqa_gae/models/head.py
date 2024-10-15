from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge, Linear
from hqa_gae.layers import get_pooling_layer

class TrainHead(nn.Module):
    def __init__(self, backbone, output_dim) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_layers = backbone.num_layers
        self.out_proj = nn.Linear(backbone.hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.backbone(x, edge_index)
        x = self.out_proj(x)
        return x

class AggrHead(nn.Module):
    def __init__(self, hidden_dim, output_dim, pooling="max") -> None:
        super().__init__()
        self.pooling = get_pooling_layer(pooling)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.pooling(x)
        x = self.out_proj(x)
        return x

class TestHead(nn.Module):
    def __init__(self, input_dim, output_dim, pooling="max") -> None:
        super().__init__()
        self.out_proj = nn.Linear(input_dim, output_dim)
        # nn.init.xavier_uniform_(self.out_proj.weight.data)
        # nn.init.zeros_(self.out_proj.bias.data)

    def forward(self, x):
        x = self.out_proj(x)
        return x