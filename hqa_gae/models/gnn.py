import torch
import torch.nn as nn
from hqa_gae.layers import (
    creat_activation_layer, 
    create_gnn_layer, 
    create_input_layer, 
    to_sparse_tensor
)
from hqa_gae.layers.gnn import MultiHeadGConv

class GNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        num_layers=2,
        dropout=0.5,
        bn=False,
        layer="gcn",
        activation="elu",
        use_node_feats=True,
        num_nodes=None,
        node_emb=None,
        concat=False,
        num_heads=1,
        heads=1,
        out_heads=1,
        hidden_mode="stack",
        output_mode="stack",
        readout=None,
        out_sigmoid=False,
        **kwargs,
    ):

        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.lins = nn.ModuleList()
        bn = nn.BatchNorm1d if bn else nn.Identity
        self.use_node_feats = use_node_feats
        self.node_emb = node_emb
        output_dim = output_dim if output_dim else hidden_dim
        self.out_sigmoid = out_sigmoid
        if 'gat' in layer:
            assert hidden_dim % heads == 0 and output_dim % out_heads == 0, "hidden_dim and output_dim must be divisible by heads"
            hidden_dim = hidden_dim // heads
            output_dim = output_dim // out_heads
        else:
            heads = out_heads = 1

        self.num_heads = num_heads
        self.readout = readout
        self.num_layers = num_layers

        if node_emb is not None and num_nodes is None:
            raise RuntimeError("Please provide the argument `num_nodes`.")

        self.input_dim, self.emb = create_input_layer(
            num_nodes, input_dim, use_node_feats=use_node_feats, node_emb=node_emb
        )

        first_channels = input_dim
        concat_dim = 0
        for i in range(num_layers):
            second_channels = output_dim if i == num_layers - 1 else hidden_dim

            if num_heads > 1:
                mode = output_mode if i == num_layers - 1 else hidden_mode
                self.convs.append(
                    MultiHeadGConv(layer, first_channels, second_channels, num_heads, mode)
                )
                conv_out_dim = second_channels * num_heads if mode == "concat" else second_channels
                self.bns.append(bn(conv_out_dim))
                
                first_channels = conv_out_dim # second_channels * num_heads if mode == "concat" or "gat" in layer else second_channels
            else:
                cur_heads = out_heads if i == num_layers - 1 else heads
                self.convs.append(create_gnn_layer(layer, first_channels, second_channels, cur_heads))
                self.bns.append(bn(second_channels * cur_heads))

                first_channels = second_channels * cur_heads
                # self.lins.append(nn.Linear(first_channels, output_dim))
            concat_dim += first_channels

        self.emb_dim = concat_dim if concat else first_channels
        self.dropout = nn.Dropout(dropout)
        self.activation = creat_activation_layer(activation)
        self.concat = concat

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

        for bn in self.bns:
            if not isinstance(bn, nn.Identity):
                bn.reset_parameters()

        if self.emb is not None:
            nn.init.xavier_uniform_(self.emb.weight)

    def create_input_feat(self, x):
        if self.use_node_feats:
            input_feat = x
            if self.node_emb:
                input_feat = torch.cat([self.emb.weight, input_feat], dim=-1)
        else:
            input_feat = self.emb.weight
        return input_feat

    def forward(self, x, edge_index, all_hidden=False):

        """
        x : (N, D) or (N, D, H)
        edge_index : (2, E)
        
        return : (N, D) or (N, D, H)
                (D, )  (D, H)
        """

        x = self.create_input_feat(x)
        edge_index = to_sparse_tensor(edge_index, x.size(0))
        out = []

        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)

        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if self.readout == "mean":
            x = x.mean(0) if x.dim() == 2 else x.mean(dim=[0, -1])
        elif self.readout == "max":
            x = x.max(0)[0] if x.dim() == 2 else x.max(dim=0)[0].max(dim=-1)[0]
        elif self.readout == "mean+max":
            x1 = x.mean(0) if x.dim() == 2 else x.mean(dim=[0, -1])
            x2 = x.max(0)[0] if x.dim() == 2 else x.max(dim=0)[0].max(dim=-1)[0]
            x = x1 + x2

        if self.out_sigmoid:
            x = x.sigmoid()

        if all_hidden:
            return torch.stack(out, dim=-1)
        elif self.concat:
            return torch.cat(out, dim=1)
        else:
            return x


    @torch.no_grad()
    def get_embedding(self, x, edge_index, mode="cat"):

        self.eval()
        assert mode in {"cat", "last"}, mode

        x = self.create_input_feat(x)
        edge_index = to_sparse_tensor(edge_index, x.size(0))
        out = []
        for i, conv in enumerate(self.convs[:-1]):
            x = self.dropout(x)
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = self.activation(x)
            out.append(x)
        x = self.dropout(x)
        x = self.convs[-1](x, edge_index)
        x = self.bns[-1](x)
        x = self.activation(x)
        out.append(x)

        if mode == "cat":
            embedding = torch.cat(out, dim=1)
        else:
            embedding = out[-1]

        return embedding
    