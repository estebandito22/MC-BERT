
import sys
#NEED TO SET FOR LOCAL DIRECTORY
sys.path.append("/home/ijh216/allennlp")

import torch
from torch import nn
from torch.nn import functional as F

from allennlp.modules.elmo import Elmo

#CURRENTLY HARD-CODED

class ElmoEmbedder(torch.nn.Module):
    def __init__(self, options_file = "/beegfs/ijh216/elmo/options.json", weights_file = "/beegfs/ijh216/elmo/weights.hdf5"):

        super(ElmoEmbedder, self).__init__()
        
        self.elmo = Elmo(options_file, weights_file, 1, requires_grad=True) 
        
        if torch.cuda.is_available():
            self.elmo = self.elmo.cuda()
            
    def forward(self, input_ids):
        
        embds = self.elmo(input_ids)['elmo_representations'][0]
        
        return embds

    def get_size(self):
        return 1024