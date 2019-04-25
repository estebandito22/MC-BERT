"""Classifier head for MCB models."""

from torch import nn
from torch.nn import CrossEntropyLoss


class ClassifierHeadModel(nn.Module):

    """Class classifier head for MCB style models."""

    def __init__(self, mcb_model, dropout=0.2, n_classes=3000):
        """Initialize SkipGramDistNet."""
        super(ClassifierHeadModel, self).__init__()
        self.mcb_model = mcb_model
        self.dropout = dropout
        self.n_classes = n_classes

        self.drop = nn.Dropout(p=self.dropout)
        self.cls = nn.Linear(self.mcb_model.output_dim, self.n_classes)

    def forward(self, vis_feats, input_ids, token_type_ids=None,
                attention_mask=None, labels=None, lm_feats=None):
        """Forward Pass."""
        # sequence_output: [batch_size, sequence_length, bert_hidden_dim]
        # pooled_output: [batch_size, bert_hidden_dim]
        lm_feats, pooled_output = self.mcb_model(
            vis_feats, input_ids, token_type_ids, attention_mask, lm_feats)

        pooled_output = self.drop(pooled_output)
        # logits: [batch_size, n_classes]
        logits = self.cls(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return lm_feats, loss
        else:
            return lm_feats, logits
