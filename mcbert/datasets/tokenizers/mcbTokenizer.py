#class for tokenizing for the MCB Original
import re
import numpy as np


class MCBTokenizer():

    def __init__(self, dict):
        self.dict = dict

    def split(self, sent):
        t_str = sent.lower()
        for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\.', r'\;', r'\`']:
            t_str = re.sub(i, '', t_str)
        for i in [r'\-', r'\/', r'\,']:
            t_str = re.sub(i, ' ', t_str)

        return t_str.split()

    def indexize(self, toks):
        return [self.dict.get_idx(tok) for tok in toks]

    def _attn_mask(self, tok):
        #this relies
        return 1 if tok != 0 else 0 

    #will probably need to modify this for text generation
    def tokenize(self, sentence, max_len=None):
        toks = self.split(sentence)

        length = len(toks)

        if max_len is not None:
            length = min(length, max_len)
            toks= toks[:max_len]
            if len(toks) < max_len:
                toks += [MCBDict.PAD] * (max_len - len(toks))

        input_ids = self.indexize(toks)
        token_type_ids = [length] * len(toks)
        attention_mask = [self._attn_mask(x) for x in input_ids]

        inp_ids = torch.tensor(input_ids).long()
        inp_len = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()

        return inp_ids, inp_len, attention_mask


class MCBDict:

    id2token = []
    token2id = {}

    glove_embds = []

    PAD = "<PAD>"
    OOV = "<OOV>"
    SOS = "<SOS>"
    EOS = "<EOS>"

    PAD_IDX = 0
    OOV_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3

    def __init__(self, metadata):
        self.id2token.append(self.PAD)
        self.id2token.append(self.OOV)
        self.id2token.append(self.SOS)
        self.id2token.append(self.EOS)
        self.glove_embds.append([0] * 300)
        self.glove_embds.append([0] * 300)
        self.glove_embds.append([0] * 300)
        self.glove_embds.append([0] * 300)

        self.token2id[self.PAD] = 0
        self.token2id[self.PAD] = 1
        self.token2id[self.PAD] = 2
        self.token2id[self.PAD] = 3

        pos = len(self.id2token)

        with open(metadata) as f:
            for l in f:
                splt = l.split()
                w = splt[0]
                glove = splt[1:]

                self.id2token.append(w)
                self.token2id[w] = pos
                self.glove_embds.append(glove)
                pos = pos +1


    def size(self):
        return len(self.id2token)

    def get_gloves(self):
        return np.array(self.glove_embds).astype(np.float)

    def get_idx(self, tok):
        if tok in self.token2id:
            return self.token2id[tok]
        else:
            return self.OOV_IDX
