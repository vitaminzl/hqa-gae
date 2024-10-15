import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pytorch_lightning as pl
from einops import rearrange
from torch import Tensor
from torch_cluster import random_walk
from torch_geometric.utils import add_self_loops, negative_sampling, dropout_edge
from hqa_gae.layers import (
    NodeMasker, PathMasker, create_node_loss_fn, create_disc_loss_fn, get_graph_readout_layer
)
from hqa_gae.models.quantizer import BaseVectorQuantizer
from hqa_gae.utils import build_optimizer
from hqa_gae.utils.test import test_link_prediction

class BaseVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def loss(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_embedding(self, *args, **kwargs):
        raise NotImplementedError()


class GAE(pl.LightningModule):
    def __init__(self, model: BaseVAE, optim: dict, optim_d=None, 
                 transductive=True, evaluator=None, split_edge=None,
                 all_edge_index=None):
        super().__init__()
        self.model = model
        self.optim = optim
        self.optim_d = optim_d

        self.transductive = transductive
        self.evaluator = evaluator
        self.automatic_optimization = True
        self.split_edge = split_edge
        self.best_score = 0.
        self.best_model = None
        self.all_edge_index = all_edge_index

    def training_step(self, batch, batch_idx):
        self.model.train()
        x, edge_index = batch.x, batch.edge_index
        x = x.clamp(0, 1)
        result = self.model(x, edge_index)
        loss, loss_log = self.model.loss(x=x, edge_index=edge_index, **result)
        self.log_dict({f"train_{k}": v  for k, v in loss_log.items()}, prog_bar=False)
        self.log("loss", loss.item(), prog_bar=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.model.eval()
        x, edge_index = batch.x, batch.edge_index
        pos_edges, neg_edges = batch.pos_edge_label_index, batch.neg_edge_label_index
            
        x = x.clamp(0, 1)
        emb = self.model.get_embedding(x, edge_index)
        # if self.split_edge:
        lp_result = test_link_prediction(emb,
                                        pos_edges=pos_edges, neg_edges=neg_edges,
                                        batch_size=65536)
        
        if lp_result["AUC"] >= self.best_score:
            self.best_score = lp_result["AUC"]
            self.best_model = copy.deepcopy(self.model)
        if dataloader_idx == 0:
            self.log_dict({f"valid_{k}": v  for k, v in lp_result.items()}, add_dataloader_idx=False, prog_bar=True)
        else:
            self.log_dict({f"test_{k}": v  for k, v in lp_result.items()}, add_dataloader_idx=False, prog_bar=True)

    def on_after_backward(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and torch.isnan(param.grad).any():
                print(f'Gradient of {name}: {param.grad}')


    def configure_optimizers(self):
        optimizer, scheduler = build_optimizer(
            self.optim.optimizer,
            self.optim.scheduler,
            self.model.parameters()
            )
        # print(scheduler is None)
        if scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer


class HQA_GAE(BaseVAE):
    def __init__(self, node_encoder: nn.Module, node_decoder: nn.Module, edge_decoder: nn.Module, 
                 vector_quantizer: BaseVectorQuantizer, vector_quantizer2: BaseVectorQuantizer=None,
                 alpha=1., beta=0.1, eta=1., drop_edge_rate=0., lambda_=2) -> None:
        super().__init__()
        self.node_encoder = node_encoder
        self.node_decoder = node_decoder
        self.edge_decoder = edge_decoder
        self.vector_quantizer = vector_quantizer
        self.vector_quantizer2 = vector_quantizer2
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.drop_edge_rate = drop_edge_rate
        self.node_rec_loss_fn = create_node_loss_fn("cosine",  lambda_=lambda_)
        self.edge_rec_loss_fn = create_disc_loss_fn("bce")

    def loss(self, *, x, x_hat, x_emb, x_quant, pos_edges_hat, neg_edges_hat, q_indices, **kwargs):

        codebook_loss = F.mse_loss(x_quant, x_emb.detach(), reduction="mean") if self.vector_quantizer.version == 2 else  x_emb.new_tensor(0)
        encoder_loss = F.mse_loss(x_quant.detach(), x_emb, reduction="mean")
        encoder_loss2 = x_emb.new_tensor(0)
        codebook_loss2 = x_emb.new_tensor(0)
        if self.vector_quantizer2:
            x_quant2 = kwargs["x_quant2"]
            codebook_loss2 = F.mse_loss(x_quant2, x_emb.detach(), reduction="mean") if self.vector_quantizer2.version == 2 else  x_emb.new_tensor(0)
            encoder_loss2 = F.mse_loss(x_quant2.detach(), x_emb, reduction="mean")

        x_rec_loss = self.node_rec_loss_fn(x_hat, x)
        edge_rec_loss = self.edge_rec_loss_fn(pos_edges_hat, neg_edges_hat)
        
        rec_loss = x_rec_loss + edge_rec_loss
        vq1_loss = codebook_loss + self.eta *  encoder_loss
        vq2_loss = (codebook_loss2 + self.eta * encoder_loss2) if self.vector_quantizer2 else x_emb.new_tensor(0)
        
        loss = rec_loss + self.alpha * vq1_loss + self.beta * vq2_loss

        loss_log = {
            "loss": loss.item(), 
            "x_rec_loss": x_rec_loss.item(), 
            "edge_rec_loss": edge_rec_loss.item(), 
            "encoder_loss": encoder_loss.item(), 
            "codebook_loss": codebook_loss.item(), 
            "encoder_loss2": encoder_loss2.item(), 
            "codebook_loss2": codebook_loss2.item(), 
        }

        return loss, {k: v for k, v in loss_log.items() if v != 0}

    def forward(self, x, edge_index, all_edge_index=None, neg_edge_index=None) -> dict[str, Tensor]:

        if self.drop_edge_rate > 0:
            edge_index, _ = dropout_edge(edge_index, p=self.drop_edge_rate, force_undirected=True)
        
        aug_edge_index, _ = add_self_loops(edge_index)
        neg_edges = negative_sampling(
            aug_edge_index,
            num_nodes=x.shape[0],
            num_neg_samples=edge_index.view(2, -1).size(1),
        ).view(2, -1)

        # x : (B, D)   edge_index : (2, E)   x_emb : (B, D) or (B, D, H)
        x_emb = self.node_encoder(x, edge_index, all_hidden=False)
        
        x_quant1, x_quant2, q_indices, q_indices1, q_indices2 = None, None, None, None, None

        x_quant1, q_indices1 = self.vector_quantizer(x_emb, h_last=True)

        # preserve gradients
        x_quant_pg1 = x_emb + (x_quant1 - x_emb).detach()

        x_quant, x_quant_pg, q_indices = x_quant1, x_quant_pg1, q_indices1

        x_quant2, q_indices2 = self.vector_quantizer2(x_quant1, h_last=True)

        # x_hat : (B, D)   A_hat : (B,) x_quant: 
        x_hat = self.node_decoder(x_quant_pg, edge_index)

        pos_edges_hat = self.edge_decoder(x_emb, edge_index)
        neg_edges_hat = self.edge_decoder(x_emb, neg_edges)

        return {
            "x_hat": x_hat, 
            "q_indices": q_indices,
            "q_indices1": q_indices1,
            "q_indices2": q_indices2,
            "x_emb": x_emb, 
            "x_quant": x_quant, 
            "x_quant1": x_quant1,
            "x_quant2": x_quant2,
            "pos_edges_hat": pos_edges_hat, 
            "neg_edges_hat": neg_edges_hat,
        }

    @torch.no_grad()
    def get_embedding(self, x, edge_index, indices=False):

        # x : (B, D)   edge_index : (2, E)   x_emb : (B, D)
        training_state = self.node_encoder.training
        self.node_encoder.train(False)
        x_emb = self.node_encoder(x, edge_index, all_hidden=False)

        embedding = x_emb 
        _, qs_indices = self.vector_quantizer.get_quants(x_emb, h_last=True)

        self.node_encoder.train(training_state)
        

        if indices:
            return embedding, qs_indices
        else:
            return embedding
        