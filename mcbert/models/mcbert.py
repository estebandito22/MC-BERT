"""MCBERT model."""

import torch
from torch import nn
import torch.nn.functional as F

from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertLayer
from pytorch_pretrained_bert.modeling import BertPooler

from mcbert.models.layers.visual.attention import AttentionMechanism
from mcbert.models.layers.composition.mcb import MCB
from compact_bilinear_pooling import CompactBilinearPooling as MCB2


class MCBertModel(nn.Module):

    """Class implementing MCBERT Model with visual attention."""

    def __init__(self, vis_feat_dim=2208, spatial_size=7, hidden_dim=768,
                 cmb_feat_dim=16000, kernel_size=3, classification=False,
                 use_external_MCB=True, use_attention=True, use_batchnorm=False,
                 lm_only=False, normalize_vis_feats=False):

        """Initialize MCBertModel."""
        super(MCBertModel, self).__init__()
        self.vis_feat_dim = vis_feat_dim
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size
        self.classification = classification
        self.use_attention = use_attention
        self.use_batchnorm = use_batchnorm
        self.lm_only = lm_only
        self.normalize_vis_feats = normalize_vis_feats

        version = "bert-base-cased"
        self.bert_model = BertModel.from_pretrained(version)
        self.bert_layer = BertLayer(self.bert_model.config)
        self.bert_pooler = BertPooler(self.bert_model.config)

        self.drop = nn.Dropout(0.1)

        if use_attention:
            self.attention = AttentionMechanism(
                self.vis_feat_dim, self.spatial_size, self.cmb_feat_dim,
                self.kernel_size, self.hidden_dim,  use_external_MCB = use_external_MCB,
                use_batchnorm=use_batchnorm)

        if use_external_MCB:
            self.compose = MCB2(self.vis_feat_dim, self.hidden_dim, self.cmb_feat_dim)
        else:
            self.compose = MCB(self.vis_feat_dim, self.hidden_dim, self.cmb_feat_dim)

        if lm_only:
            self.output_dim = self.hidden_dim
        else:
            self.output_dim = self.cmb_feat_dim

    def signed_sqrt(self, x):
        return torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))

    def bert_forward(self, input_ids, token_type_ids, attention_mask):

        bert_sequence_output, orig_pooled_output = self.bert_model(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False)

        return bert_sequence_output, orig_pooled_output.unsqueeze(1)

    def forward(self, vis_feats, input_ids, token_type_ids=None,
                attention_mask=None, lm_feats = None):
        """Forward Pass."""

        if lm_feats is None:

            # sequence_output: [batch_size, sequence_length, hidden_dim]
            # pooled_output: [batch_size, hidden_dim]
            bert_sequence_output, orig_pooled_output = self.bert_forward(
                input_ids, token_type_ids, attention_mask)

            lm_feats = orig_pooled_output
        else:
            if lm_feats[0,0,0] == 0:
                print("Recieved a zero vector???", lm_feats)

        if self.lm_only:
            # batch_size x hidden_dim
            pooled_output  = lm_feats.squeeze(1)
        else:
            if self.classification:
                # batch_size x 1 x hidden_dim
                cls_sequence_output = lm_feats
                cls_vis_feats = vis_feats[:, 0, :, :].unsqueeze(1)
                if self.use_attention:
                    if self.normalize_vis_feats: vis_feats = vis_feats / torch.sqrt((vis_feats**2).sum())
                    cls_vis_feats = self.attention(cls_vis_feats, cls_sequence_output)
                else:
                    cls_vis_feats = cls_vis_feats.view(cls_vis_feats.shape[0], 1, self.vis_feat_dim, -1).mean(-1)
                    if self.normalize_vis_feats: vis_feats = vis_feats / torch.sqrt((vis_feats**2).sum())

                # batch_size x 1 x cmb_feat_dim
                cmb_feats = self.compose(
                    cls_vis_feats, cls_sequence_output)

                # signsqrt and l2 normalize
                cmb_feats = self.signed_sqrt(cmb_feats)
                cmb_feats = F.normalize(cmb_feats, p=2, dim=2)
                cmb_feats = self.drop(cmb_feats)

                # batch_size x cmb_feat_dim
                pooled_output = cmb_feats.squeeze(1)
            else:
                # batch_size x sequence_length x hidden_dim
                sequence_vis_feats = self.attention(
                    vis_feats, bert_sequence_output)

                # batch_size x seqlen x hidden_dim
                sequence_cmb_feats = self.compose(
                    sequence_vis_feats, bert_sequence_output )

                # see  https://github.com/huggingface/pytorch-pretrained-BERT/blob/
                # 7cc35c31040d8bdfcadc274c087d6a73c2036210/pytorch_pretrained_bert/
                # modeling.py#L639
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = extended_attention_mask.float()
                extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

                sequence_output = self.bert_layer(
                    sequence_cmb_feats, extended_attention_mask)
                pooled_output = self.bert_pooler(sequence_cmb_feats)
                # hack to complete graph of original Bert model
                pooled_output = pooled_output + orig_pooled_output
                # hack to work for pretraining
                lm_feats = sequence_output



        return lm_feats, pooled_output
