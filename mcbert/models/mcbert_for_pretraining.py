"""BERT Visual Memory model."""

from torch import nn
from torch.nn import CrossEntropyLoss

from pytorch_pretrained_bert import BertForPreTraining

from mcbert.models.mcbert import MCBertModel


class MCBertForPretrainingModel(nn.Module):

    """Class implementing MCBERT model for unsupervised pre-training."""

    def __init__(self, vis_feat_dim=2208, spatial_size=7, bert_hidden_dim=768,
                 cmb_feat_dim=16000, kernel_size=3):
        """Initialize SkipGramDistNet."""
        super(MCBertForPretrainingModel, self).__init__()
        self.vis_feat_dim = vis_feat_dim
        self.spatial_size = spatial_size
        self.bert_hidden_dim = bert_hidden_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size

        self.mcbert_model = MCBertModel(
            vis_feat_dim=vis_feat_dim, spatial_size=spatial_size,
            bert_hidden_dim=bert_hidden_dim, cmb_feat_dim=cmb_feat_dim,
            kernel_size=kernel_size)

        version = "bert-base-cased"
        bert_model = BertForPreTraining.from_pretrained(version)
        self.cls = bert_model.cls

        self.vocab_size = bert_model.config.vocab_size

    def forward(self, vis_feats, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None):
        """Forward Pass."""
        # sequence_output: [batch_size, sequence_length, bert_hidden_dim]
        # pooled_output: [batch_size, bert_hidden_dim]
        sequence_output, pooled_output = self.mcbert_model(
            vis_feats, input_ids, token_type_ids, attention_mask)

        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.vocab_size),
                masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(
                    -1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss

        return prediction_scores, seq_relationship_score
