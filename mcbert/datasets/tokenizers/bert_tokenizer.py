
from pytorch_pretrained_bert import BertTokenizer as OrigBertTokenizer
import torch

class BertTokenizer():

    version = "bert-base-cased"
    tokenizer = OrigBertTokenizer.from_pretrained(version)

    def _attn_mask(self, tok):
        #this relies
        return 1 if tok != 0 else 0 

    def tokenize(self, sentence, max_len):

        tokenized = ['[CLS]'] + self.tokenizer.tokenize(sentence)

        if max_len is not None:
            tokenized= tokenized[:max_len]
            if len(tokenized) < max_len:
                tokenized += ['[PAD]'] * (max_len - len(tokenized))

        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        token_type_ids = [0] * len(tokenized)
        attention_mask = [self._attn_mask(x) for x in input_ids]

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()

        return  input_ids, token_type_ids, attention_mask
