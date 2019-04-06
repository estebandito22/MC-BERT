"""Class for loading pretraining dataset for BERT dataset."""

import numpy as np

import torch
from torch.utils.data import Dataset

class VQADataset(Dataset):

    """Class to load VQA dataset."""

    def __init__(self, metadata, tokenizer, n_classes, max_sent_len=64):
        """
        Initialize VQA.
        Args
            metadata : dataframe, ordered fields img_path, idxseq
        """
        self.metadata = metadata
        self.max_sent_len = max_sent_len
        self.tokenizer = tokenizer
        self.n_classes = n_classes

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

    def __len__(self):
        """Return length of dataset."""
        return len(self.metadata)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        #label
        label = min(self.n_classes - 1, self.metadata.iat[i, 2])

        # question_id
        qid = torch.tensor(self.metadata.iat[i, 3]).long()

        #Sentence Features
        sentence = self.metadata.iat[i, 1]
        input_ids, token_type_ids, attention_mask = self.tokenizer.tokenize(sentence, self.max_sent_len)
        
        #Visual Features
        vis_feats_path = self.metadata.iat[i, -1]
        vis_feats = torch.load(vis_feats_path)
        vis_feats = vis_feats.unsqueeze(0).repeat(input_ids.size(0), 1, 1, 1)
        vis_feats = vis_feats.detach()

        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'labels': label,
                'qids' : qid,
                'attention_mask': attention_mask,
                'vis_feats': vis_feats}
