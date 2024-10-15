import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

def get_pooling_layer(type="max", dim=0):
    if type == "max":
        return lambda x: torch.max(x, dim=dim).values
    else:
        return lambda x: torch.mean(x, dim=dim)
    
def get_graph_readout_layer(type):
    if type == "mean":
        return global_mean_pool
    elif type == "max":
        return global_max_pool
    elif type == "add":
        return global_add_pool