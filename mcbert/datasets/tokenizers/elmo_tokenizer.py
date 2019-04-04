
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
        tokens = [tok.text for i, tok in enumerate(self.tokenizer.batch_tokenize([sentence])[0]) if i < max_len]
        
        inp_ids =  batch_to_ids(tokens)
        
        inp_len = torch.tensor(inp_ids.size(0)).repeat(inp_ids.size(0))

        return  inp_ids, inp_len, None
