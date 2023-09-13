# coding=utf-8
import os
from turtle import forward
import numpy as np

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    # norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    # X = torch.div(X, norm)
    X = X / (torch.norm(X,p=1,dim=dim,keepdim=True)+eps)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    # norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    # X = torch.div(X, norm)
    X = X / (torch.norm(X,p=2,dim=dim,keepdim=True)+eps)
    return X

def cosine_similarity(x1, x2, dim=-1):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2)).squeeze(dim=-1)

# calculate the mask according to the given lens
def get_mask(lens):
    """
    :param lens: length of the sequence
    :return: 
    """
    batch = lens.shape[0]
    max_l = int(lens.max())
    mask = torch.arange(max_l).expand(batch, max_l).to(lens.device)
    mask = (mask<lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(lens.device)
    return mask

def get_padding_mask(lens):
    """
    :param lens: length of the sequence
    :return: 
    """
    batch = lens.shape[0]
    max_l = int(lens.max())
    mask = torch.arange(max_l).expand(batch, max_l).to(lens.device)
    mask = (mask>=lens.long().unsqueeze(dim=1)).to(lens.device)
    return mask

# calculate the fine-grained similarity according to the given images and captions
def get_fgsims(imgs, caps):
    bi, n_r, embi = imgs.shape
    bc, n_w, embc = caps.shape
    imgs = imgs.reshape(bi*n_r, embi)
    caps = caps.reshape(bc*n_w, embc).t()
    sims = torch.matmul(imgs,caps)
    sims = sims.reshape(bi, n_r, bc, n_w).permute(0,2,1,3)
    return sims


# calculate the mask of fine-grained similarity according to the given images length and captions length
def get_fgmask(img_lens, cap_lens):
    bi = img_lens.shape[0]
    bc = cap_lens.shape[0]
    max_r = int(img_lens.max())
    max_w = int(cap_lens.max())

    mask_i = torch.arange(max_r).expand(bi, max_r).to(img_lens.device)
    mask_i = (mask_i<img_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(img_lens.device)
    mask_i = mask_i.reshape(bi*max_r,1)

    mask_c = torch.arange(max_w).expand(bc,max_w).to(cap_lens.device)
    mask_c = (mask_c<cap_lens.long().unsqueeze(dim=1)).float().unsqueeze(-1).to(cap_lens.device)
    mask_c = mask_c.reshape(bc*max_w,1).t()

    mask = torch.matmul(mask_i,mask_c).reshape(bi, max_r, bc, max_w).permute(0,2,1,3)
    return mask

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B*N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8):
        super(SelfAttention, self).__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)

    def attention(self, x: torch.Tensor, lens: torch.Tensor=None):
        mask = get_padding_mask(lens).squeeze() if lens is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=mask)[0]

    def forward(self, x: torch.Tensor, lens: torch.Tensor=None):
        return x + self.attention(self.ln(x),lens)


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_head: int = 8):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

def SCAN_attention(query, context, smooth=9):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch, queryL, sourceL
    attn = F.softmax(attn*smooth, dim=2)

    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    weightedContext = l2norm(weightedContext, dim=-1)

    return weightedContext, attn
