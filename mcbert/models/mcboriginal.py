"""MCB_Orig model."""

from torch import nn
import torch

from mcbert.models.layers.visual.attention import AttentionMechanism
from mcbert.models.layers.composition.mcb import MCB
from compact_bilinear_pooling import CompactBilinearPooling as MCB2

class MCBOriginalModel(nn.Module):

    """Class implementing MCB Model with visual attention."""

    def __init__(self, embedder, vis_feat_dim=2208, spatial_size=7,  hidden_dim = 2208,
                 cmb_feat_dim=16000, kernel_size=3, bidirectional=False, 
                 classification=True,  use_external_MCB=True, use_attention=True ):


        """Initialize MCBertModel."""
        super(MCBOriginalModel, self).__init__()
        self.vis_feat_dim = vis_feat_dim
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size
        self.use_attention = use_attention

        #hint to whatever head uses us - 
        self.output_dim = cmb_feat_dim

        #each layer (or direction) gets its own part
        lstm_hidden_dim = int(hidden_dim / 2 / (2 if bidirectional else 1))

        self.embedder = embedder

        # TODO: initialize weights weight_filler=dict(type='uniform',min=-0.08,max=0.08)
        self.lstm = nn.LSTM(embedder.get_size(), num_layers=2, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=bidirectional, dropout=0.3)
        self.drop = nn.Dropout(0.3)

        if use_attention:
            self.attention = AttentionMechanism(
                self.vis_feat_dim, self.spatial_size, self.cmb_feat_dim,
                self.kernel_size, self.hidden_dim,  use_external_MCB=use_external_MCB)

        if use_external_MCB:
            self.compose = MCB2(self.vis_feat_dim, self.hidden_dim, self.cmb_feat_dim)
        else:
            self.compose = MCB(self.vis_feat_dim, self.hidden_dim, self.cmb_feat_dim)

        # signed sqrt

    def forward(self, vis_feats, input_ids, token_type_ids=None,
                attention_mask=None):
        """Forward Pass."""

        #remeber our batchsize
        bs = input_ids.shape[0]

        #bit of a hack, but we pass the length through in the token_type_ids
        #just need one per example
        lengths = token_type_ids[:,[0]].squeeze(-1)

        inpt = self.embedder(input_ids)

        #get our sort order and sort our inputs
        order = torch.argsort(lengths, descending=True)
        restore = torch.argsort(order)
        lengths = lengths[order]
        inpt = inpt[order]

        pack = torch.nn.utils.rnn.pack_padded_sequence(inpt, lengths, batch_first=True)
        lout, (hlayers, _) = self.lstm(pack)

        # sequence_output: [batch_size, sequence_length, bert_hidden_dim]
        # pooled_output: [batch_size, bert_hidden_dim]

        hlayers = hlayers.transpose(0, 1)
        hlayers = hlayers[restore]
        hlayers = self.drop(hlayers)

        orig_pooled_output = hlayers.view(bs, -1).unsqueeze(1)

        #some tasks require the visual features to be tiled, but we just need a single copy here
        vis_feats = vis_feats[:, 0, :, :].unsqueeze(1)

        # batch_size x sequence_length x hidden_dim
        if self.use_attention:
            sequence_vis_feats = self.attention(vis_feats, orig_pooled_output)
        else:
            sequence_vis_feats = vis_feats.view(bs, 1, self.vis_feat_dim, -1).mean(-1)

        # batch_size x seqlen x cmb_feat_dim
        sequence_cmb_feats = self.compose(
            orig_pooled_output, sequence_vis_feats)

        pooled_output = sequence_cmb_feats.squeeze(1)#[:,-1,:]

        return sequence_cmb_feats, pooled_output
