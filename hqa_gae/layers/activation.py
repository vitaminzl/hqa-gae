import torch.nn as nn

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise ValueError("Unknown activation")