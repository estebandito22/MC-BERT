"""MCB_Orig model."""

from torch import nn
import torch

from mcbert.models.layers.visual.attention import AttentionMechanism
from mcbert.models.layers.composition.mcb import MCB


class MCBLMOnlyModel(nn.Module):

    """Class implementing MCB Model with visual attention."""

    def __init__(self, embedder, vis_feat_dim=2208, spatial_size=7,  hidden_dim = 2208,
                 cmb_feat_dim=16000, kernel_size=3, bidirectional=False, classification = True ):


        """Initialize MCBertModel."""
        super(MCBLMOnlyModel, self).__init__()
        self.vis_feat_dim = vis_feat_dim
        self.spatial_size = spatial_size
        self.hidden_dim = hidden_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size

        #hint to whatever head uses us - 
        self.output_dim = hidden_dim

        #each layer (or direction) gets its own part
        lstm_hidden_dim = int(hidden_dim / 2 / (2 if bidirectional else 1))

        self.embedder = embedder

        self.lstm = nn.LSTM(embedder.get_size(), num_layers=2, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=bidirectional, dropout=0.3) #weight_filler=dict(type='uniform',min=-0.08,max=0.08)
        self.drop = nn.Dropout(0.3)


    def forward(self, vis_feats, input_ids, token_type_ids=None,
                attention_mask=None):
        """Forward Pass."""

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

        #print("hlayers:", hlayers.shape)
        lstmhids = []
        for i in range(hlayers.shape[1]):
            lstmhids.append(hlayers[:,i,:].unsqueeze(1))
        orig_pooled_output = torch.cat(lstmhids, dim=2)
   
        orig_pooled_output = orig_pooled_output.squeeze(1)

        return _, orig_pooled_output
