"""Class for loading pretraining dataset for BERT dataset."""

import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split


class VQADataset(Dataset):

    """Class to load Pinterest dataset."""

    def __init__(self, metadata, tokenizer, n_classes, split='train', max_sent_len=64):
        """
        Initialize PinterestDataset.
        Args
            metadata : dataframe, ordered fields img_path, idxseq
        """
        self.metadata = metadata
        self.split = split
        self.max_sent_len = max_sent_len
        self.tokenizer = tokenizer
        self._train_test_split()
        self.n_classes = n_classes

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

    def _attn_mask(self, tok):
        #this relies
        return 1 if tok != 0 else 0 

    def __len__(self):
        """Return length of dataset."""
        return len(self.metadata)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        vis_feats_path = self.metadata.iat[i, -1]
        sentence = self.metadata.iat[i, 1]
        label = min(self.n_classes - 1, self.metadata.iat[i, 2])

        # tokenize, add any special characters, and return indexes
        input_ids, token_type_ids = self.tokenizer.tokenize(sentence, self.max_sent_len)

        attention_mask = [self._attn_mask(x) for x in input_ids]

        input_ids = torch.tensor(input_ids).long()
        token_type_ids = torch.tensor(token_type_ids).long()
        attention_mask = torch.tensor(attention_mask).long()

        vis_feats = torch.load(vis_feats_path)
        vis_feats = vis_feats.unsqueeze(0).repeat(
            input_ids.size(0), 1, 1, 1)
        vis_feats = vis_feats.detach()

        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'labels': label,
                'attention_mask': attention_mask,
                'vis_feats': vis_feats}
