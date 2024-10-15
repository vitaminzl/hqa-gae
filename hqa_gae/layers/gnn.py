from einops import rearrange
import torch
import torch.nn as nn
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn import (
    Linear,
    GCNConv,
    SAGEConv,
    GATConv,
    GINConv,
    GATv2Conv,
    global_add_pool,
    global_mean_pool,
    global_max_pool
)

def to_sparse_tensor(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)


def create_input_layer(num_nodes, num_node_feats,
                       use_node_feats=True, node_emb=None):
    emb = None
    if use_node_feats:
        input_dim = num_node_feats
        if node_emb:
            emb = torch.nn.Embedding(num_nodes, node_emb)
            input_dim += node_emb
    else:
        emb = torch.nn.Embedding(num_nodes, node_emb)
        input_dim = node_emb
    return input_dim, emb

def create_gnn_layer(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(first_channels, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(first_channels, second_channels, heads=heads)
    elif name == "linear":
        layer = LinearGraphWrapper(first_channels, second_channels)
    else:
        raise ValueError(name)
    
    return layer

class MultiHeadGConv(nn.Module):
    def __init__(self, name, first_channels, second_channels, heads, mode="stack"):
        """
        
        """
        super().__init__()
        self.heads = heads
        self.name = name
        # if "gat" in name and mode == "concat":
        #     self.convs = create_gnn_layer(name, first_channels, second_channels, heads)
        # else:
        self.convs = nn.ModuleList([ 
                create_gnn_layer(name, first_channels, second_channels, 1)
                for _ in range(heads)
                ])
        self.mode = mode

    def forward(self, x: Tensor, edge_index: Tensor):
        """
        x : (N, D) or (N, D, H)
        edge_index : (2, E)
        """
        # if isinstance(self.convs, nn.ModuleList):
        out_list = []
        for i, conv in enumerate(self.convs):
            if x.dim() == 3:
                assert x.shape[-1] == len(self.convs)
                out_list.append(conv(x[..., i], edge_index))
            else:
                out_list.append(conv(x, edge_index))
        out: Tensor = rearrange(out_list, "H N D -> N D H") # (N, D, H)

        if self.mode == "stack":
            pass
        elif self.mode == "concat":
            out = rearrange(out, "N D H -> N (H D)") # (N, H*D)
        elif self.mode == "mean":
            out = out.mean(dim=-1)
        elif self.mode == "max":
            out = out.max(dim=-1)[0]
        else:
            raise ValueError(self.mode)

        return out



class LinearGraphWrapper(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.linear(x)