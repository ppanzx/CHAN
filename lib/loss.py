import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss (max-margin based)
    """

    def __init__(self, opt=None, margin=0.2, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        if opt is not None:
            self.opt = opt
            self.margin = opt.margin
            self.max_violation = opt.max_violation
        else:
            self.margin = margin
            self.max_violation = max_violation

    def max_violation_on(self):
        self.max_violation = True
        print('Use VSE++ objective.')

    def max_violation_off(self):
        self.max_violation = False
        print('Use VSE0 objective.')

    def forward(self, sims):
        # compute image-sentence score matrix
        # sims = get_sim(im, s)
        diagonal = sims.diag().view(sims.size(0), 1)
        d1 = diagonal.expand_as(sims)
        d2 = diagonal.t().expand_as(sims)

        # compare every diagonal score to sims in its column
        # caption retrieval
        cost_s = (self.margin + sims - d1).clamp(min=0)
        # compare every diagonal score to sims in its row
        # image retrieval
        cost_im = (self.margin + sims - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(sims.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

class InfoNCELoss(nn.Module):
    """
    Compute InfoNCELoss loss
    """
    def __init__(self, temperature=0.01, margin=0):
        super(InfoNCELoss, self).__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, sims):
        ## cost of image retrieval
        img_ret = sims-sims.diag().expand_as(sims).t()+self.margin
        img_ret[torch.eye(sims.size(0))>.5] = 0
        cost_im = torch.log(torch.sum(torch.exp(img_ret/self.temperature),dim=1))

        ## cost of text retrieval
        txt_ret = sims-sims.diag().expand_as(sims)+self.margin
        txt_ret[torch.eye(sims.size(0))>.5] = 0
        cost_s = torch.log(torch.sum(torch.exp(txt_ret/self.temperature),dim=0))

        return cost_s.mean() + cost_im.mean()

    def max_violation_on(self):
        return 

    def max_violation_off(self):
        return

def get_criterion(criterion,opt,**args):
    if criterion=="ContrastiveLoss":
        return ContrastiveLoss(margin=opt.margin)
    elif criterion=="InfoNCELoss":
        return InfoNCELoss(temperature=opt.temperature,
                            margin=opt.margin)
    else:
        raise ValueError("Unknown criterion type: {}".format(criterion))