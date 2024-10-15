import math
from typing import List, Optional, Sequence, Tuple, Union
from einops import rearrange
from torch import Tensor
from hqa_gae.layers.embedding import NormalEmbedding
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseVectorQuantizer(nn.Module):
    def __init__(self, codebook_size: int, code_dim: int, distance_type: str, 
                 padding_idx: int, version: int, t0=1., prob_decay=0., eps=1e-5) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.padding_idx = padding_idx
        assert distance_type in ["square", "cosine", None]
        self.distance_type = distance_type
        self.version = version
        self.prob_decay = prob_decay
        self.eps = eps
        self.temperature = t0

    def get_q_indices(self, z: Tensor):
        """
        z : (..., D)
        
        """
        shape = z.shape
        assert shape[-1] == self.code_dim, f"Last dimension of z must be {self.code_dim}, but got {shape[-1]}"
        z = z.contiguous().view(-1, self.code_dim)
        
        emb_mtx: Tensor = self.get_codebook_weight().data
        
        # (codebook_size -1, code_dim)
        if self.padding_idx >= 0:
            idx = torch.arange(self.codebook_size, device=z.device, dtype=torch.int64)
            idx = torch.concat((idx[:self.padding_idx], idx[self.padding_idx+1:]), dim=0)
            emb_mtx = emb_mtx[idx]

        if self.distance_type == "cosine":
            cos_sim = F.normalize(z, dim=-1) @ F.normalize(emb_mtx.T, dim=0)
            if self.prob_decay > 0 and self.training:
                q_probs = F.softmax(cos_sim / self.temperature, dim=1)
                q_indices = torch.multinomial(q_probs, num_samples=1)
                # q_indices = F.softmax(cos_sim, dim=1)
            else:
                q_indices = torch.argmax(cos_sim, dim=1)

        elif self.distance_type == "square":
            distance = (z.unsqueeze(1) - emb_mtx.unsqueeze(0)).pow(2).sum(-1)
            if self.prob_decay > 0 and self.training:
                q_probs = F.softmax(-distance / self.temperature, dim=1)
                q_indices = torch.multinomial(q_probs, num_samples=1)
                # q_indices = F.softmax(-distance, dim=1)
            else:
                q_indices = torch.argmin(distance, dim=1)
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")
        
        if self.padding_idx >= 0:
            q_indices = idx[q_indices]
        q_indices = q_indices.view(*shape[:-1], -1)

        self.temperature = max(self.prob_decay * self.temperature, self.eps) #if self.temperature > self.eps else self.temperature

        return q_indices
    
    def get_codebook_weight(self) -> Tensor:
        raise NotImplementedError()
    
    def embed_code(self, *args, **kwargs):
        raise NotImplementedError()
    
    def get_codebook(self, *args, **kwargs):
        raise NotImplementedError()
    
    @torch.no_grad()
    def get_quants(self, z: Tensor, h_last=False) -> Union[Tensor, Tensor]:  # z_BChw is the feature from inp_img_no_grad
        if h_last and z.dim() == 3:
            z = rearrange(z, "B D H -> B H D")

        q_indices = self.get_q_indices(z)
        embedding = self.embed_code(q_indices)
        
        if h_last and embedding.dim() == 3:
            embedding = rearrange(embedding, "B H D -> B D H")
        return embedding, q_indices


class VectorQuantizer(BaseVectorQuantizer):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, codebook_size, code_dim, distance_type="square", padding_idx=0, decay=0.99, eps=1e-5, 
        prob_decay=False, t0=1.
    ):
        super().__init__(codebook_size=codebook_size, code_dim=code_dim, 
                         distance_type=distance_type, padding_idx=padding_idx,
                         prob_decay=prob_decay, t0=t0, version=2)
        self.decay = decay
        self.eps = eps

        # embedding = torch.randn(codebook_size, code_dim).uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)
        embedding = torch.randn(codebook_size, code_dim).uniform_(-1.0 / self.codebook_size, 1.0 / self.codebook_size)

        self.register_buffer("embedding", embedding)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", embedding.clone())


    def embed_code(self, embed_id):
        if embed_id.shape[-1] == 1:
            embed_id = embed_id.squeeze(-1)
            return F.embedding(embed_id, self.embedding, self.padding_idx)
        else:
            return embed_id @ self.embedding
        
    def get_codebook_weight(self):
        return self.embedding.data

    def get_codebook(self):
        self.embedding: torch.Tensor
        codebook =  nn.Embedding(self.embedding.size(0), self.embedding.size(1),
                                 padding_idx=self.padding_idx, device=self.embedding.device)
        codebook.weight.data[:self.padding_idx] = self.embedding.data[:self.padding_idx]
        codebook.weight.data[self.padding_idx+1:] = self.embedding.data[self.padding_idx+1:]
        return codebook

    def forward(self, z: Tensor, h_last=False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        z : (B, D) or (B, H, D), (B, D, H) if h_last=True
        """
        if h_last and z.dim() == 3:
            z = rearrange(z, "B D H -> B H D")

        assert z.shape[-1] == self.code_dim, f"Last dimension of z must be {self.code_dim}, but got {z.shape[-1]}"
         # (B, D) or (B * H, D)
        flatten = z.reshape(-1, self.code_dim)
        q_indices = self.get_q_indices(flatten)

        if q_indices.shape[-1] == 1:
            q_indices = q_indices.squeeze(-1)
            # (B, N) or (B * H, N)
            embed_onehot = F.one_hot(q_indices, self.codebook_size).type(flatten.dtype)
        else:
            embed_onehot = q_indices

        q_indices = q_indices.view(*z.shape[:-1], -1)
        embedding = self.embed_code(q_indices)

        if self.training:
            
            # embed_onehot_sum (N, )
            embed_onehot_sum = embed_onehot.sum(0)
            # embed_sum (N, D)
            embed_sum = embed_onehot.transpose(0, 1) @ flatten
            # update cluster_size (N, )
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            # update embed_avg (N, D)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            # cluster_size (N, )
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
            )
            # embed_normalized (N, D)
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
            embed_normalized[0] = embed_normalized.new_zeros(embed_normalized[0].shape)
            self.embedding.data.copy_(embed_normalized)

        if h_last and embedding.dim() == 3:
            embedding = rearrange(embedding, "B H D -> B D H")

        return embedding, q_indices
    


class VectorQuantizer2(BaseVectorQuantizer):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - codebook_size : number of embeddings
    - code_dim : dimension of embedding
    """

    def __init__(self, codebook_size, code_dim, distance_type="square", padding_idx=0, prob_decay=0, t0=1., **kwargs):
        super().__init__(codebook_size=codebook_size, code_dim=code_dim, 
                         distance_type=distance_type, padding_idx=padding_idx,
                         version=1, prob_decay=prob_decay, t0=t0)
        self.codebook = NormalEmbedding(self.codebook_size, self.code_dim,)


    def embed_code(self, q_indices):
        if q_indices.shape[-1] == 1:
            q_indices = q_indices.squeeze(-1)
            return self.codebook(q_indices) # + self.pe(q_indices)
        else:
            return q_indices @ self.codebook.weight

    def get_codebook(self):
        return self.codebook
    
    def get_codebook_weight(self):
        return self.codebook.get_weight() # + self.pe.pe.data

    def forward(self, z: Tensor, h_last=False):
        """
        idx (B, ) or (B, H)
        return (B, H, D) or (B, D)
        """
        if h_last and z.dim() == 3:
            z = rearrange(z, "B D H -> B H D")
        q_indices = self.get_q_indices(z)
        embedding = self.embed_code(q_indices)
        if h_last and embedding.dim() == 3:
            embedding = rearrange(embedding, "B H D -> B D H")
        return embedding, q_indices

