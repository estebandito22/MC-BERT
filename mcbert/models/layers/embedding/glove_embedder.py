
import torch
from torch import nn
from torch.nn import functional as F
from mcbert.datasets.tokenizers.mcbtokenizer import MCBDict

class GloveEmbedder(torch.nn.Module):

    def __init__(self,vocab_file, embd_dim ):
        super(GloveEmbedder, self).__init__()

        self.embd_dim = embd_dim

        #create the dictionary, also has our glove embeddings
        dict = MCBDict(metadata=vocab_file)
        vocab_size = dict.size()

        # weight_filler=dict(type='uniform',min=-0.08,max=0.08))
        self.embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=0)

        self.glove = nn.Embedding(vocab_size, embd_dim, padding_idx=0)

        # override the word embeddings with pre-trained
        self.glove.weight = nn.Parameter(torch.tensor(dict.get_gloves()).float())

        # build mask  (Or, especially if we don't need EOS/SOS, just make OOV random
        self.embeddings_mask = torch.zeros(vocab_size, requires_grad=False).float()
        self.embeddings_mask[0:4] = 1
        self.embeddings_mask.resize_(vocab_size, 1)

        if torch.cuda.is_available():
            self.embeddings_mask = self.embeddings_mask.cuda()

        # mask pretrained embeddings
        self.glove.weight.register_hook(
            lambda grad: grad * self.embeddings_mask)

    def forward(self, input_ids):

        # combine our two embeddings
        embds = F.tanh(self.embedding(input_ids))
        gloves = self.glove(input_ids)
        inpt = torch.cat((embds, gloves), dim=2)

        return inpt

    def get_size(self):
        return self.embd_dim * 2