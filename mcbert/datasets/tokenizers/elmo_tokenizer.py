
import torch

from allennlp.modules.elmo import batch_to_ids
from allennlp.data.tokenizers import WordTokenizer



class ElmoTokenizer():

    version = "elmo-base-cased"
    tokenizer = WordTokenizer
    
    def __init__(self):
        self.tokenizer = WordTokenizer()
        return

    def tokenize(self, sentence, max_len):
        
        #Will only work with one sentence
        tokens = ['<bos>'] + \
            [tok.text for i, tok in enumerate(self.tokenizer.batch_tokenize([sentence])[0]) if i < max_len - 2] + \
                ['<eos>']
        
        inp_ids =  batch_to_ids([tokens])
        inp_ids = torch.cat([inp_ids, torch.zeros(1, max_len-inp_ids.size(1), 50).long()], 1)
        
        inp_len = torch.tensor(inp_ids.size(1)).repeat(inp_ids.size(1))
        
        #Ignore
        attn_mask = torch.tensor(inp_ids.size(1)).repeat(inp_ids.size(1))

        return  inp_ids, inp_len, attn_mask
