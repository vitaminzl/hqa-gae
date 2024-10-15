import torch.nn as nn
from hqa_gae.layers import (
    creat_activation_layer, 
)

class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, input_dim, hidden_dim, output_dim=1,
        num_layers=2, dropout=0.5, activation='relu', num_heads=1, input_pooling=None
    ):

        super().__init__()

        if num_layers is None or hidden_dim is None:
            self.mlps = nn.Linear(input_dim, output_dim)
        else:
            self.mlps = nn.ModuleList()
            self.input_pooling = input_pooling

            input_dim = input_dim // num_heads if input_pooling != "concat" else input_dim 
            for i in range(num_layers):
                first_channels = input_dim if i == 0 else hidden_dim
                second_channels = output_dim if i == num_layers - 1 else hidden_dim
                self.mlps.append(nn.Linear(first_channels, second_channels))

            self.dropout = nn.Dropout(dropout)
            self.activation = creat_activation_layer(activation)

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True, reduction=False):
        """
        z : N
        """
        
        if not isinstance(self.mlps, nn.ModuleList):
            x = self.mlps(z)
        else:
            x = z[edge_index[0]] * z[edge_index[1]]

            for i, mlp in enumerate(self.mlps[:-1]):
                x = self.dropout(x)
                x = mlp(x)
                x = self.activation(x)

            x = self.mlps[-1](x)

        if sigmoid or not self.training:
            x = x.sigmoid()
        else:
            x = x
        
        return x