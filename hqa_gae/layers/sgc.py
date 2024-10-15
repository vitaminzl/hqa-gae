import torch
import math
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing, Linear
from torch_geometric.utils import add_self_loops, degree
from einops import rearrange

class SGCLayer(MessagePassing):
    """
    SGC Layer类用于实现SGC图卷积层。

    Args:
        num_layers (int): 图卷积层的数量。
        layer_agg (str): 图卷积层的聚合方式，可选值为 "sum"、"mean"、"concat"、"concat_flatten"、"last"。
        reverse (bool, optional): 是否反转图卷积层的输出顺序。默认为False。
        add_self_loops (bool, optional): 是否在图中添加自环边。默认为True。
        batch_first (bool, optional): 是否将批次维度放在第一维，当且仅当 layer_agg 为 concat 时起作用。默认为False。
    """

    def __init__(self, num_layers, layer_agg, reverse=False, add_self_loops=True, batch_first=False, norm_func=None):
        super().__init__(node_dim=0, aggr='add')
        self.num_layers = num_layers
        self.layer_agg = layer_agg
        self.reverse = reverse
        self.add_self_loops = add_self_loops
        self.batch_first = batch_first
        self.norm_func = norm_func

    def forward(self, x, edge_index, emb=False):
        """
        前向传播函数，用于计算图卷积层的输出。

        Args:
            x (Tensor): 输入特征张量，形状为 [N, input_dim]。
            edge_index (LongTensor): 边索引张量，形状为 [2, E]。
            emb (bool, optional): 是否返回每个图卷积层的输出。默认为False。

        Returns:
            Tensor: 图卷积层的输出张量，形状根据聚合方式不同而有所不同。
        """
        if self.add_self_loops:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # using normalization of GCN, D^{-1/2}AD^{-1/2}
        row, col = edge_index
        
        deg_col = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt_col = deg_col.pow(-0.5)
        deg_inv_sqrt_col[deg_inv_sqrt_col == float('inf')] = 0

        deg_row = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt_row = deg_row.pow(-0.5)
        deg_inv_sqrt_row[deg_inv_sqrt_row == float('inf')] = 0
        
        norm = deg_inv_sqrt_row[row] * deg_inv_sqrt_col[col]

        # Start propagating messages.
        emb_ls = [x]
        for l in range(self.num_layers):
            x = self.propagate(edge_index, x=x, norm=norm)
            emb_ls.append(x)

        # rearange results
        if self.layer_agg == "sum":
            out = torch.einsum("lbd->bd", emb_ls)
        elif self.layer_agg == "mean":
            out = torch.einsum("lbd->bd", emb_ls) / len(emb_ls)
        elif self.layer_agg == "stack":
            if self.reverse:
                emb_ls = emb_ls[::-1]
            out = torch.stack(emb_ls, dim=0)
            if self.batch_first:
                out = rearrange(emb_ls, "l b d -> l b d")
        elif self.layer_agg == "concat_flatten":
            if self.reverse:
                emb_ls = emb_ls[::-1]
            out = rearrange(emb_ls, "l b d -> b (l d)")
        elif self.layer_agg == "last":
            out = emb_ls[-1]
        else:
            raise NotImplementedError(f"Unknown layer aggregation type {self.layer_agg}")

        if self.norm_func:
            out = self.norm_func(out)
        return out

    def message(self, x_j, norm):
        """
        消息传递函数，用于计算每个节点接收到的消息。

        Args:
            x_j (Tensor): 邻居节点的特征张量，形状为 [E, output_dim]。
            norm (Tensor): 归一化系数张量，形状为 [E, 1]。

        Returns:
            Tensor: 经过归一化处理后的消息张量，形状为 [E, output_dim]。
        """

        return norm.view(-1, 1) * x_j