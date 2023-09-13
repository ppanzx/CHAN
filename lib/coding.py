# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules import get_mask, get_fgsims, get_fgmask, l2norm, cosine_similarity, SCAN_attention

EPS = 1e-8 # epsilon 
MASK = -1 # padding value

# Visual Hard Assignment Coding
class VHACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        return sims

# Texual Hard Assignment Coding
class THACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-1)[0]
        return sims

class VSACoding(nn.Module):
    def __init__(self,temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)

        # calculate attention
        sims = sims / self.temperature

        sims = torch.softmax(sims.masked_fill(mask==0, -torch.inf),dim=-1) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,caps) # Bi x Bt x K x D
        sims = torch.mul(sims.permute(1,0,2,3),imgs).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bi x Bt x K

        mask = get_mask(img_lens).permute(0,2,1).repeat(1,cap_lens.size(0),1)
        sims = sims.masked_fill(mask==0, -1)
        return sims

class T2ICrossAttentionPool(nn.Module):
    def __init__(self,smooth=9):
        super().__init__()
        self.labmda = smooth

    def forward(self, imgs, caps, img_lens, cap_lens):
        return self.xattn_score_t2i(imgs,caps,cap_lens)

    def xattn_score_t2i(self, images, captions, cap_lens, return_attn=False):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        attentions = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = int(cap_lens[i].item())
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d)
                attn: (n_image, n_region, n_word)
            """
            if return_attn:
                weiContext,attn = SCAN_attention(cap_i_expand, images,self.labmda)
                attentions.append(attn)
            else:
                weiContext,_ = SCAN_attention(cap_i_expand, images,self.labmda)
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_word)
            col_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
            col_sim = col_sim.mean(dim=1, keepdim=True)
            similarities.append(col_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        if return_attn:return torch.cat(attentions, 0)
        else:return similarities

# max pooling
class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims = sims.max(dim=-1)[0]
        return sims

# mean pooling
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        lens = (sims!=MASK).sum(dim=-1)
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)/lens
        return sims

# sum pooling
class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)
        return sims

# log-sum-exp pooling
class LSEPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        sims = torch.logsumexp(sims/self.temperature,dim=-1)
        return sims

# softmax pooling
class SoftmaxPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = -torch.inf
        weight = torch.softmax(sims/self.temperature,dim=-1)
        sims = (weight*sims).sum(dim=-1)
        return sims

def get_coding(coding_type, **args):
    alpha = args["opt"].alpha
    if coding_type=="VHACoding":
        return VHACoding()
    elif coding_type=="THACoding":
        return THACoding()
    elif coding_type=="VSACoding":
        return VSACoding(alpha)
    else:
        raise ValueError("Unknown coding type: {}".format(coding_type))

def get_pooling(pooling_type, **args):
    belta = args["opt"].belta
    if pooling_type=="MaxPooling":
        return MaxPooling()
    elif pooling_type=="MeanPooling":
        return MeanPooling()
    elif pooling_type=="SumPooling":
        return SumPooling()
    elif pooling_type=="SoftmaxPooling":
        return SoftmaxPooling(belta)
    elif pooling_type=="LSEPooling":
        return LSEPooling(belta)
    else:
        raise ValueError("Unknown pooling type: {}".format(pooling_type))
