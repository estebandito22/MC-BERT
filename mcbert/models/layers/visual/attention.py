"""Class for conv attention from https://arxiv.org/pdf/1606.01847.pdf."""

from torch import nn
import torch.nn.functional as F

from mcbert.models.layers.composition.mcb import MCB
from compact_bilinear_pooling import CompactBilinearPooling as MCB2


class AttentionMechanism(nn.Module):

    """AttentionMechanism."""

    def __init__(self, feat_dim, spatial_size, cmb_feat_dim, kernel_size,
                 hidden_size, use_external_MCB=True):
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

        self.conv1 = nn.Conv2d(
            in_channels=self.cmb_feat_dim, out_channels=512,
            kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=1,
            kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        if use_external_MCB:
            self.compose_func = MCB2(self.feat_dim, self.hidden_size, self.cmb_feat_dim)
        else:
            self.compose_func = MCB(self.feat_dim, self.hidden_size, self.cmb_feat_dim)

    def forward(self, vis_feats, txt_feats):
        """Forward Pass."""
        # vis_feats: batch_size x feat_dim x height x width
        # txt_feats: batch_size x seqlen x hidden_size x height x width
        bs, seqlen, hidden_size = txt_feats.size()
        txt_feats = txt_feats.unsqueeze(-1).unsqueeze(-1).repeat(
            1, 1, 1, self.spatial_size, self.spatial_size)

        bs, seqlen, vis_feat_dim, height, width = vis_feats.size()

        #bs x (seq_len * height * width) x feat_dim
        #TODO: note this currently only works if seq_len == 1
        if use_external_MCB:
            txt_feats = txt_feats.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, hidden_size)
            vis_feats_transform = vis_feats.permute(0, 1, 3, 4, 2).contiguous().view(bs, -1, vis_feat_dim)
        else:
            vis_feats_transform = vis_feats

        # outputs batch_size x seqlen x cmb_feat_dim x height x width
        x = self.compose_func(vis_feats_transform, txt_feats)
        if use_external_MCB: x = x.permute(0, 2, 1)
        x = x.contiguous()
        x = x.view(bs * seqlen, self.cmb_feat_dim, height, width)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        # batch_size x seqlen x 1 x height x width
        x = x.view(bs, seqlen, 1, -1)
        attn_weights = F.softmax(x, dim=-1).view(bs, seqlen, 1, height, width)

        # batch_size x seqlen x hidden_size x height x width
        attn_vect = vis_feats * attn_weights

        # batch_size x seqlen x hidden_size
        return attn_vect.view(bs, seqlen, vis_feat_dim, -1).sum(dim=-1)
