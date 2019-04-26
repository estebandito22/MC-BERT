
import torch
from torch import nn
from torch.nn import functional as F
from mcbert.datasets.tokenizers.mcb_tokenizer import MCBDict

class BasicEmbedder(torch.nn.Module):

    def __init__(self, vocab_file, embd_dim ):
        super(BasicEmbedder, self).__init__()

        self.embd_dim = embd_dim

        #create the dictionary, also has our glove embeddings
        dict = MCBDict(metadata=vocab_file)
        vocab_size = dict.size()

        # weight_filler=dict(type='uniform',min=-0.08,max=0.08))
        self.embedding = nn.Embedding(vocab_size, embd_dim, padding_idx=0)

    def forward(self, input_ids):

        embds = F.tanh(self.embedding(input_ids))

        return embds

    def get_size(self):
        return self.embd_dim
