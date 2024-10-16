from .head import TestHead, TrainHead, AggrHead
from .gae import GAE, HQA_GAE
from .quantizer import VectorQuantizer, VectorQuantizer2
from .gnn import GNN
from .mlp import EdgeDecoder
from torch import nn, Tensor
from torch_geometric.nn import MLP


def create_gae(config, dataset):

    def create_vq(vq_type, **kwargs):
        if vq_type == 1:
            return VectorQuantizer(
                **kwargs
            )
        elif vq_type == 2:
            return VectorQuantizer2(
                **kwargs
            )
    
    encoder = GNN(
        dataset.num_features,
        layer=config.gnn.node_encoder,
        **config.gnn
    )

    vq = create_vq(
        vq_type=config.vq_type,
        code_dim=encoder.emb_dim,
        codebook_size=config.q1_size,
        t0=config.t0,
        prob_decay=config.temp_decay,
        )

    vq2 = create_vq(
        vq_type=config.vq_type,
        code_dim=encoder.emb_dim,
        codebook_size=config.q2_size,
        )
    
    dec = config.gnn.to_dict()
    dec.pop("output_dim")
    if "concat" in dec:
        dec.pop("concat")
    if "dec_dropout" in dec:
        dec["dropout"] = dec["dec_dropout"]


    node_decoder = GNN(
        input_dim=encoder.emb_dim if vq is None else vq.code_dim,
        output_dim=dataset.num_features,
        heads=dec.get("dec_heads", 4),
        layer="gat",
        **dec   
    )

    edge_decoder = EdgeDecoder(
        input_dim=encoder.emb_dim,
        **config.mlp
    )

    
    gae = HQA_GAE(
        encoder, 
        node_decoder, 
        edge_decoder, 
        vq,
        vq2,
        alpha=config.get("alpha", 1.),
        beta=config.get("beta", 0.1),
        lambda_=config.get("lambda_", 2),
        drop_edge_rate=config.get("drop_edge_rate", 0.),
    )
    
    return gae
    