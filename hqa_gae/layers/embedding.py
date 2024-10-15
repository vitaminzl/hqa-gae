import torch.nn as nn
import torch.nn.functional as F


class NormalEmbedding(nn.Embedding):
    def __init__(self, codebook_size, code_dim) -> None:
        super().__init__(codebook_size, code_dim)
        self.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

    def get_weight(self):
        return self.weight

    def forward(self, x):
        return super().forward(x)
