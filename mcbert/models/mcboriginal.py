"""MCBERT model."""

from torch import nn
import torch.nn.functional as F
import torch


from mcbert.models.layers.visual.attention import AttentionMechanism
from mcbert.models.layers.composition.mcb import MCB
from util.mcbtokenizer import MCBDict

class MCBOriginalModel(nn.Module):

    """Class implementing MCBERT Model with visual attention."""

    #questions - what does the bert tokenizer return, what are token_type_ids
    #where do I want to put loading the dictionary

    def __init__(self, vocab_file, vis_feat_dim=2208, spatial_size=7, embd_dim=300, bert_hidden_dim = 2048,
                 cmb_feat_dim=16000, kernel_size=3 ):
        """Initialize MCBertModel."""
        super(MCBOriginalModel, self).__init__()
        self.vis_feat_dim = vis_feat_dim
        self.spatial_size = spatial_size
        self.hidden_dim = bert_hidden_dim
        self.cmb_feat_dim = cmb_feat_dim
        self.kernel_size = kernel_size

        #probably want to do this elsewhere and pass in but...
        dict = MCBDict(metadata=vocab_file)
        vocab_size = dict.size()

        # override the word embeddings with pre-trained
        self.glove = nn.Embedding(vocab_size, embd_dim, padding_idx=0)
        self.glove.weight = nn.Parameter(dict.get_gloves())

        # build mask  (Or, especially if we don't need EOS/SOS, just make OOV random
        self.embeddings_mask = torch.zeros(vocab_size).float()
        self.embeddings_mask[0] = 1
        self.embeddings_mask[1] = 1
        self.embeddings_mask[2] = 1
        self.embeddings_mask[3] = 1
        self.embeddings_mask.requires_grad = False
        self.embeddings_mask.resize_(vocab_size, 1)

        # mask pretrained embeddings
        self.glove.weight.register_hook(
            lambda grad: grad * self.embeddings_mask)

        self.embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=0) #weight_filler=dict(type='uniform',min=-0.08,max=0.08))
        self.layer1 = nn.LSTM(embd_dim*2, hidden_size=1024) #weight_filler=dict(type='uniform',min=-0.08,max=0.08)
        self.drop1 = nn.Dropout(0.3)
        self.layer2 = nn.LSTM(embd_dim*2, hidden_size=1024)  # weight_filler=dict(type='uniform',min=-0.08,max=0.08)
        self.drop1 = nn.Dropout(0.3)

        self.attention = AttentionMechanism(
            self.vis_feat_dim, self.spatial_size, self.cmb_feat_dim,
            self.kernel_size, self.bert_hidden_dim)

        self.compose = MCB(self.bert_hidden_dim, self.bert_hidden_dim)

    def forward(self, vis_feats, input_ids, token_type_ids=None,
                attention_mask=None):
        """Forward Pass."""
        embds = F.tanh(self.embedding(input_ids))
        gloves = self.glove(input_ids)
        embds = torch.cat(embds, gloves)
        # concat with 300dim glove embedding
        l1out, hlayers1 = self.layer1(embds)
        l1out = self.drop1(l1out)
        hlayers1 = self.drop1(hlayers1)
        l2out, hlayers2 = self.layer2(l1out)
        l2out = self.drop2(l2out)
        hlayers2 = self.drop2(hlayers2)

        # sequence_output: [batch_size, sequence_length, bert_hidden_dim]
        # pooled_output: [batch_size, bert_hidden_dim]
        bert_sequence_output = torch.cat(hlayers1, hlayers2)
        orig_pooled_output = torch.cat(l1out, l2out)

        # batch_size x sequence_length x bert_hidden_dim
        sequence_vis_feats = self.attention(vis_feats, bert_sequence_output)

        # batch_size x seqlen x cmb_feat_dim
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
