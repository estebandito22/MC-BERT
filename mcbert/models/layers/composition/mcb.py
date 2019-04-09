"""Compact Bilinear Pooling."""

import numpy as np
import torch
from torch import nn

# batch x seqlen x feat_dim x spatial size x spatial size
# vis_feats = torch.randn((1, 2, 4, 2, 2))
# txt_feats = torch.randn((1, 2, 4, 2, 2))
#
#
# 
# vis_h = torch.randint(0, 8, (4,))
# vis_h
# vis_h = vis_h.view(1, 1, 4, 1, 1)
# vis_h = vis_h.repeat(1, 2, 1, 2, 2)
# vis_h.size()
#
#
# txt_h = torch.randint(0, 8, (4,))
# txt_h = txt_h.view(1, 1, 4, 1, 1)
# txt_h = txt_h.repeat(1, 2, 1, 2, 2)
# txt_h.size()
#
# vis_s = torch.from_numpy(np.random.choice([-1,1], (4,))).float()
# vis_s
# vis_s = vis_s.view(1, 1, 4, 1, 1)
# vis_s = vis_s.repeat(1, 2, 1, 2, 2)
# vis_s.size()
#
#
# txt_s = torch.from_numpy(np.random.choice([-1,1], (4,))).float()
# txt_s = txt_s.view(1, 1, 4, 1, 1)
# txt_s = txt_s.repeat(1, 2, 1, 2, 2)
# txt_s.size()
#
# vis_y = torch.zeros(1, 2, 8, 2, 2)
# txt_y = torch.zeros(1, 2, 8, 2, 2)
#
# vis_feats_psi = vis_y.scatter_add_(dim=2, index=vis_h, src=vis_feats * vis_s)
# txt_feats_psi = txt_y.scatter_add_(dim=2, index=txt_h, src=txt_feats * txt_s)
#
#
# vis_feats_psi = vis_feats_psi.permute(0, 1, 3, 4, 2).contiguous()
# txt_feats_psi = vis_feats_psi.permute(0, 1, 3, 4, 2).contiguous()
#
# vis_feats_psi = vis_feats_psi.view(-1, 8)
# vis_feats_dft = torch.rfft(vis_feats_psi, 1, normalized=False)
# vis_feats_dft.shape
#
# txt_feats_psi = txt_feats_psi.view(-1, 8)
# txt_feats_dft = torch.rfft(txt_feats_psi, 1, normalized=False)
# txt_feats_dft.shape
#
# cmb_feats_dft = vis_feats_dft * txt_feats_dft
# cmb_feats_dft.shape
#
# cmb_feats = torch.irfft(cmb_feats_dft, 1, normalized=False, signal_sizes=(8,))
# cmb_feats.shape
#
# cmb_feats = cmb_feats.view(1, 2, 2, 2, 8)
# cmb_feats = cmb_feats.permute(0, 1, 4, 2, 3).contiguous()
# cmb_feats.shape


class MCB(nn.Module):

    """Class to concatenate and project word embeddings."""

    def __init__(self, feat_dim, cmb_feat_dim):
        """
        Initialize ConcatProject.

        Args
        ----
            feat_dim : int, dim of feature vector.
            cmb_feat_dim : int, dim of combined feature vectors (i.e. output).

        """
        super(MCB, self).__init__()
        self.feat_dim = feat_dim
        self.cmb_feat_dim = cmb_feat_dim

        np.random.seed(1234)
        torch.manual_seed(1234)

        self.vis_h = torch.randint(0, self.cmb_feat_dim, (self.feat_dim,))
        self.vis_h = self.vis_h.view(1, 1, self.feat_dim, 1, 1)
        self.txt_h = torch.randint(0, self.cmb_feat_dim, (self.feat_dim,))
        self.txt_h = self.txt_h.view(1, 1, self.feat_dim, 1, 1)

        self.vis_s = torch.from_numpy(
            np.random.choice([-1, 1], (self.feat_dim,))).float()
        self.vis_s = self.vis_s.view(1, 1, self.feat_dim, 1, 1)
        self.txt_s = torch.from_numpy(
            np.random.choice([-1, 1], (self.feat_dim,))).float()
        self.txt_s = self.txt_s.view(1, 1, self.feat_dim, 1, 1)

        if torch.cuda.is_available():
            self.vis_h = self.vis_h.cuda()
            self.txt_h = self.txt_h.cuda()
            self.vis_s = self.vis_s.cuda()
            self.txt_s = self.txt_s.cuda()

        self.drop = nn.Dropout(0.1)

    def forward(self, vis_feats, txt_feats):
        """Forward Pass."""
        # batch size x seqlen x feat_dim (x height x width)
        if vis_feats.dim() == 5:
            bs, seqlen, _, height, width = vis_feats.size()
            squeeze = False
        elif vis_feats.dim() == 3:
            squeeze = True
            vis_feats = vis_feats.unsqueeze(-1).unsqueeze(-1)
            txt_feats = txt_feats.unsqueeze(-1).unsqueeze(-1)
            bs, seqlen, _, height, width = vis_feats.size()

        vis_h = self.vis_h.repeat(bs, seqlen, 1, height, width)
        txt_h = self.txt_h.repeat(bs, seqlen, 1, height, width)

        vis_s = self.vis_s.repeat(bs, seqlen, 1, height, width)
        txt_s = self.txt_s.repeat(bs, seqlen, 1, height, width)

        vis_y = torch.zeros(bs, seqlen, self.cmb_feat_dim, height, width)
        txt_y = torch.zeros(bs, seqlen, self.cmb_feat_dim, height, width)

        if torch.cuda.is_available():
            vis_y = vis_y.cuda()
            txt_y = txt_y.cuda()

        vis_feats_psi = vis_y.scatter_add_(
            dim=2, index=vis_h, src=vis_feats * vis_s)
        txt_feats_psi = txt_y.scatter_add_(
            dim=2, index=txt_h, src=txt_feats * txt_s)

        vis_feats_psi = vis_feats_psi.permute(0, 1, 3, 4, 2).contiguous()
        txt_feats_psi = txt_feats_psi.permute(0, 1, 3, 4, 2).contiguous()

        vis_feats_psi = vis_feats_psi.view(-1, self.cmb_feat_dim)
        vis_feats_dft = torch.rfft(vis_feats_psi, 1, normalized=False)

        txt_feats_psi = txt_feats_psi.view(-1, self.cmb_feat_dim)
        txt_feats_dft = torch.rfft(txt_feats_psi, 1, normalized=False)

        cmb_feats_dft = vis_feats_dft * txt_feats_dft

        cmb_feats = torch.irfft(
            cmb_feats_dft, 1, normalized=False,
            signal_sizes=(self.cmb_feat_dim,))

        cmb_feats = cmb_feats.view(
            bs, seqlen, height, width, self.cmb_feat_dim)
        cmb_feats = cmb_feats.permute(0, 1, 4, 2, 3).contiguous()

        # batch size x seqlen x cmb_feat_dim (x height x width)

        if squeeze:
            cmb_feats = cmb_feats.view(bs, seqlen, self.cmb_feat_dim)

        cmb_feats = self.signed_sqrt(cmb_feats)
        cmb_feats = nn.functional.normalize(cmb_feats, p=2, dim=2)
        cmb_feats = self.drop(cmb_feats)

        return cmb_feats

    def signed_sqrt(self, x):
        return torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
