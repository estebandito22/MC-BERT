"""MCBERT model."""

import torch
from torch import nn

from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.modeling import BertLayer
from pytorch_pretrained_bert.modeling import BertPooler

from mcbert.models.layers.visual.attention import AttentionMechanism
from mcbert.models.layers.composition.mcb import MCB


class MCBertModel(nn.Module):

    """Class implementing MCBERT Model with visual attention."""

    def __init__(self, vis_feat_dim=2208, spatial_size=7, hidden_dim=768,
                 cmb_feat_dim=16000, kernel_size=3, classification=False):
        """Initialize MCBertModel."""
        super(MCBertModel, self).__init__()
        self.vis_feat_dim = vis_feat_dim
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size
        self.classification = classification

        version = "bert-base-cased"
        self.bert_model = BertModel.from_pretrained(version)
        self.bert_layer = BertLayer(self.bert_model.config)
        self.bert_pooler = BertPooler(self.bert_model.config)

        self.attention = AttentionMechanism(
            self.vis_feat_dim, self.spatial_size, self.cmb_feat_dim,
            self.kernel_size, self.hidden_dim)

        self.compose = MCB(self.hidden_dim, self.hidden_dim)

    def forward(self, vis_feats, input_ids, token_type_ids=None,
                attention_mask=None):
        """Forward Pass."""
        # sequence_output: [batch_size, sequence_length, hidden_dim]
        # pooled_output: [batch_size, hidden_dim]
        bert_sequence_output, orig_pooled_output = self.bert_model(
            input_ids, token_type_ids, attention_mask,
            output_all_encoded_layers=False)

        if self.classification:
            # batch_size x 1 x hidden_dim
            cls_sequence_output = bert_sequence_output[:, 0, :].unsqueeze(1)
            cls_vis_feats = vis_feats[:, 0, :, :].unsqueeze(1)
            cls_vis_feats = self.attention(cls_vis_feats, cls_sequence_output)

            # batch_size x 1 x cmb_feat_dim
            cls_cmb_feats = self.compose(
                cls_sequence_output, cls_vis_feats)

            # batch_size x seqlen x hidden_dim
            sequence_cmb_feats = torch.cat(
                [cls_cmb_feats, bert_sequence_output[:, 1:, :]], dim=1)
        else:
            # batch_size x sequence_length x hidden_dim
            sequence_vis_feats = self.attention(
                vis_feats, bert_sequence_output)

            # batch_size x seqlen x hidden_dim
            sequence_cmb_feats = self.compose(
                bert_sequence_output, sequence_vis_feats)

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

        return sequence_output, pooled_output
