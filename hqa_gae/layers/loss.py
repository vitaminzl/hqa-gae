from functools import partial
import torch
import torch.nn.functional as F

def zero_fn(*args, **kwargs):
    return torch.Tensor(0)

def create_disc_loss_fn(loss_type, **kwargs):
    if loss_type == "bce":
        edge_rec_loss_fn = partial(ce_loss, **kwargs)
    elif loss_type == "auc":
        edge_rec_loss_fn = partial(auc_loss, **kwargs)
    elif loss_type == "info_nce":
        edge_rec_loss_fn = partial(info_nce_loss, **kwargs)
    elif loss_type == "log_rank":
        edge_rec_loss_fn = partial(log_rank_loss, **kwargs)
    elif loss_type == "hinge_auc":
        edge_rec_loss_fn = partial(hinge_auc_loss, **kwargs)
    elif loss_type is None:
        return zero_fn
    else:
        raise ValueError(loss_type)
    return edge_rec_loss_fn

def create_node_loss_fn(loss_type, **kwargs):
    if loss_type == "bce":
        node_rec_loss_fn = partial(F.binary_cross_entropy, **kwargs)
    elif loss_type == "mse":
        node_rec_loss_fn = partial(F.mse_loss, **kwargs)
    elif loss_type == "cosine":
        node_rec_loss_fn = partial(scaled_cosine_loss, **kwargs)
    elif loss_type is None:
        return zero_fn
    else:
        raise ValueError(loss_type)
    return node_rec_loss_fn

def scaled_cosine_loss(source, target, lambda_=2):
    # source: (N, D), target: (N, D)  -> (N, )
    cosine_similarty = F.cosine_similarity(source, target, dim=-1)
    return  torch.mean((1 - cosine_similarty) ** lambda_)

def auc_loss(pos_out, neg_out):
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out, num_neg=1):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def ce_loss(pos_out, neg_out, with_logits=False):
    if with_logits:
        pos_loss = F.binary_cross_entropy_with_logits(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy_with_logits(neg_out, torch.zeros_like(neg_out))        
    else:
        pos_loss = F.binary_cross_entropy(pos_out, torch.ones_like(pos_out))
        neg_loss = F.binary_cross_entropy(neg_out, torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def info_nce_loss(pos_out, neg_out):
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()