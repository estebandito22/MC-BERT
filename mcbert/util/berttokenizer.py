
from pytorch_pretrained_bert import BertTokenizer as OrigBertTokenizer


class BertTokenizer():

    version = "bert-base-cased"
    tokenizer = OrigBertTokenizer.from_pretrained(version)


    def tokenize(self, sentence, max_len):

        tokenized = ['[CLS]'] + [self.tokenizer.tokenize(x) for x in sentence.split(" ")]

        if max_len is not None:
            tokenized= tokenized[:max_len]
            if len(tokenized) < max_len:
                tokenized += ['[PAD]'] * (max_len - len(tokenized))

                input_ids = self.tokenizer.convert_tokens_to_ids(tokenized)

        token_type_ids = [0] * len(tokenized)

        return  input_ids, token_type_ids