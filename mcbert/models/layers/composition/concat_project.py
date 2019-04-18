"""Placeholder to combine visual and text features."""

import torch
from torch import nn


class ConcatProject(nn.Module):

    """Class to concatenate and project word embeddings."""

    def __init__(self, vis_feat_dim, feat_dim, cmb_feat_dim):
        """
        Initialize ConcatProject.

        Args
        ----
            feat_dim : int, dim of feature vector.
            cmb_feat_dim : int, dim of combined feature vectors (i.e. output).

        """
        super(ConcatProject, self).__init__()
        self.vis_feat_dim = vis_feat_dim
        self.feat_dim = feat_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.proj = nn.Linear(
            self.vis_feat_dim + self.feat_dim, self.cmb_feat_dim)
        self.drop = nn.Dropout(0.1)

    def signed_sqrt(self, x):
        return torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))

    def forward(self, vis_feats, txt_feats):
        """Forward Pass."""
        # batch_size x seqlen x feat_dim (x height x width)
        x = torch.cat([vis_feats, txt_feats], dim=2)

        if x.dim() == 5:
            x = x.permute(0, 1, 3, 4, 2)

        x = self.proj(x)

        if x.dim() == 5:
            x.permute(0, 1, 4, 2, 3)

        x = self.signed_sqrt(x)
        x = nn.functional.normalize(x, p=2, dim=2)
        x = self.drop(x)

        # batch size x * x cmb_feat_dim (x height x width)
        return x
