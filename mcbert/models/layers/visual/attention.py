"""Class for conv attention from https://arxiv.org/pdf/1606.01847.pdf."""

from torch import nn
import torch.nn.functional as F

from mcbert.models.layers.composition.mcb import MCB


class AttentionMechanism(nn.Module):

    """AttentionMechanism."""

    def __init__(self, feat_dim, spatial_size, cmb_feat_dim, kernel_size,
                 hidden_size):
        """
        Initialize AttentionMechanism.

        Args
        ----
            feat_dim : int, dimension of incoming visual features.
            spatial_size : int, spatial size of incoming visual features.
            cmb_feat_dim : int, dimension of the combined embeddings.
            kernel_size : int, kernel size to use in conv.
            hidden_size : int, hidden size corresponding to BERT features.

        """
        super(AttentionMechanism, self).__init__()
        self.feat_dim = feat_dim
        self.spatial_size = spatial_size
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size
        self.hidden_size = hidden_size
        assert self.kernel_size % 2 == 1, "kernel_size must be odd."

        self.project = nn.Conv2d(
            in_channels=self.feat_dim, out_channels=self.hidden_size,
            kernel_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=self.cmb_feat_dim, out_channels=512,
            kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=1,
            kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.compose_func = MCB(self.hidden_size, self.cmb_feat_dim)

    def forward(self, vis_feats, txt_feats):
        """Forward Pass."""
        # vis_feats: batch_size x feat_dim x height x width
        # txt_feats: batch_size x seqlen x hidden_size x height x width
        bs, seqlen, hidden_size = txt_feats.size()
        txt_feats = txt_feats.unsqueeze(-1).unsqueeze(-1).expand(
            bs, seqlen, hidden_size, self.spatial_size, self.spatial_size)

        bs, seqlen, vis_feat_dim, height, width = vis_feats.size()
        vis_feats = self.project(
            vis_feats.view(bs * seqlen, vis_feat_dim, height, width)).view(
                bs, seqlen, hidden_size, height, width)

        # outputs batch_size x seqlen x cmb_feat_dim x height x width
        x = self.compose_func(vis_feats, txt_feats)
        x = x.view(bs * seqlen, self.cmb_feat_dim, height, width)
        x = self.conv1(x)
        x = self.conv2(x)

        # batch_size x seqlen x 1 x height x width
        x = x.view(bs, seqlen, 1, -1)
        attn_weights = F.softmax(x, dim=-1).view(bs, seqlen, 1, height, width)

        # batch_size x seqlen x hidden_size x height x width
        attn_vect = vis_feats * attn_weights

        # batch_size x seqlen x hidden_size
        return attn_vect.view(bs, seqlen, hidden_size, -1).sum(dim=-1)
