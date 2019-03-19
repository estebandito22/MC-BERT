"""Class for loading pretraining dataset for BERT dataset."""

import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer


class PinterestPretrainDataset(Dataset):

    """Class to load Pinterest dataset."""

    def __init__(self, metadata, vocab, split='train', max_sent_len=64):
        """
        Initialize PinterestDataset.

        Args
            metadata : dataframe, ordered fields img_path, idxseq
        """
        self.metadata = metadata
        self.split = split
        self.max_sent_len = max_sent_len
        self.vocab = vocab
        version = "bert-base-cased"
        self.tokenizer = BertTokenizer.from_pretrained(version)
        self._train_test_split()

    def _train_test_split(self):
        X_train, X_val_test = train_test_split(
            self.metadata, test_size=0.1, random_state=10)
        X_val, X_test = train_test_split(
            X_val_test, test_size=0.095/0.1, random_state=10)

        if self.split == 'train':
            self.metadata = X_train
        elif self.split == 'val':
            self.metadata = X_val
        elif self.split == 'test':
            self.metadata = X_test
        else:
            raise ValueError("split must be 'train', 'val' or 'test'.")

    def get_batches(self, k=10):
        """Return index batches of inputs."""
        indexes = [x for x in range(len(self))]
        np.random.shuffle(indexes)
        s = 0
        size = int(np.ceil(len(indexes) / k))
        batches = []
        while s < len(indexes):
            batches += [indexes[s:s + size]]
            s = s + size
        return batches

    def _mask_sentence(self, sentence, rate=0.15):
        uniform = np.random.uniform(0, 1, len(sentence))
        mask_prob = 0.8 * rate
        rand_prob = 0.9 * rate

        tgt_sentence = np.where(uniform > rate, '[PAD]', sentence)

        # mask out words
        sentence = np.where(uniform <= mask_prob, '[MASK]', sentence)

        # replace with random words
        rand_cond = (uniform > mask_prob) & (uniform <= rand_prob)
        rand_words = np.random.choice(self.vocab, (len(sentence),))
        sentence = np.where(rand_cond, rand_words, sentence)

        # remaining words are left alone

        return [list(sentence), list(tgt_sentence)]

    def _pad_sentence(self, sentence, max_len=None):
        if max_len is None:
            max_len = self.max_sent_len
        sentence = sentence[:max_len]
        if len(sentence) < max_len:
            sentence += ['[PAD]'] * (max_len - len(sentence))
        return sentence

    def _pad_sentence_pair(self, sentence_pair):
        l0 = len(sentence_pair[0])
        l1 = len(sentence_pair[1])
        if l0 + l1 > self.max_sent_len:
            if l0 > self.max_sent_len // 2:
                sentence_pair[0] = sentence_pair[0][:self.max_sent_len // 2]
                if l1 > self.max_sent_len // 2:
                    sentence_pair[1] = \
                        sentence_pair[1][:self.max_sent_len // 2]
                else:
                    sentence_pair[1] = self._pad_sentence(
                        sentence_pair[1], self.max_sent_len // 2)
            else:
                sentence_pair[1] = \
                    sentence_pair[1][:self.max_sent_len - l0]
        else:
            sentence_pair[1] = self._pad_sentence(
                sentence_pair[1], self.max_sent_len - l0)
        return sentence_pair

    @staticmethod
    def _attn_mask(sentence):
        return [1 if x != '[PAD]' else 0 for x in sentence]

    def _prepare_sentence_data(self, sentence_pair):
        sentence_pair = [self.tokenizer.tokenize(x) for x in sentence_pair]
        sentence_pair = [self._mask_sentence(x) for x in sentence_pair]
        sentence_pair, tgt_sentence_pair = zip(*sentence_pair)

        sentence_pair = [x + ['[SEP]'] for x in sentence_pair]
        sentence_pair[0] = ['[CLS]'] + sentence_pair[0]
        sentence_pair = self._pad_sentence_pair(sentence_pair)

        tgt_sentence_pair = [x + ['[SEP]'] for x in tgt_sentence_pair]
        tgt_sentence_pair = [['[CLS]'] + tgt_sentence_pair[0],
                             tgt_sentence_pair[1]]
        tgt_sentence_pair = self._pad_sentence_pair(tgt_sentence_pair)

        token_type_ids = [[i] * len(x) for i, x in enumerate(sentence_pair)]
        attention_mask = [self._attn_mask(x) for x in sentence_pair]

        sentence_pair = list(np.concatenate(sentence_pair))
        tgt_sentence_pair = list(np.concatenate(tgt_sentence_pair))
        token_type_ids = list(np.concatenate(token_type_ids))
        attention_mask = list(np.concatenate(attention_mask))

        return sentence_pair, tgt_sentence_pair, token_type_ids, attention_mask

    def _neg_sample(self, sentences):
        part1 = np.random.choice(sentences, (1,))
        rand_idx = np.random.randint(0, len(self))
        rand_sentence = eval(self.metadata.iat[rand_idx, 1])
        part2 = np.random.choice(rand_sentence, (1,))

        return list(part1) + list(part2)

    @staticmethod
    def _pos_sample(sentences):
        return np.random.choice(sentences, (2,), replace=False)

    def __len__(self):
        """Return length of dataset."""
        return len(self.metadata)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        vis_feats_path = self.metadata.iat[i, 3]
        sentences = eval(self.metadata.iat[i, 1])

        if len(sentences) >= 2:
            if np.random.uniform() > 0.5:
                sentence_pair = self._pos_sample(sentences)
                sentence_pair, tgt_sentence_pair, token_type_ids, \
                    attention_mask = self._prepare_sentence_data(sentence_pair)
                next_sentence_label = [0]
            else:
                sentence_pair = self._neg_sample(sentences)
                sentence_pair, tgt_sentence_pair, token_type_ids, \
                    attention_mask = self._prepare_sentence_data(sentence_pair)
                next_sentence_label = [1]
        else:
            sentence_pair = self._neg_sample(sentences)
            sentence_pair, tgt_sentence_pair, token_type_ids, \
                attention_mask = self._prepare_sentence_data(sentence_pair)
            next_sentence_label = [1]

        # add special tokens
        input_ids = self.tokenizer.convert_tokens_to_ids(sentence_pair)
        tgt_input_ids = self.tokenizer.convert_tokens_to_ids(tgt_sentence_pair)

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()
        masked_lm_labels = torch.tensor(tgt_input_ids).long()
        next_sentence_label = torch.tensor(next_sentence_label).long()

        masked_lm_labels.masked_fill_(
            torch.tensor(tgt_input_ids == 0), torch.tensor(-1))

        vis_feats = torch.load(vis_feats_path)
        vis_feats = vis_feats.unsqueeze(0).repeat(
            input_ids.size(0), 1, 1, 1)
        vis_feats = vis_feats.detach()

        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'masked_lm_labels': masked_lm_labels,
                'next_sentence_label': next_sentence_label,
                'vis_feats': vis_feats}
