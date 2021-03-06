"""Class for loading pretraining dataset for BERT dataset."""

import os
import hashlib
import numpy as np

import torch
from torch.utils.data import Dataset


class VQADataset(Dataset):

    """Class to load VQA dataset."""

    def __init__(self, metadata, tokenizer, n_classes, max_sent_len=64,
                 hidden_size=None):
        """
        Initialize VQA.
        Args
            metadata : dataframe, ordered fields img_path, idxseq
        """
        self.metadata = metadata
        self.max_sent_len = max_sent_len
        self.tokenizer = tokenizer
        self.n_classes = n_classes
        self.hidden_size = hidden_size
        self.input_ids_dict = {}
        self.input_tensors_dict = {}

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

    def sample(self, p=1.0, seed=0):
        """Sample a subset of the data."""
        np.random.seed(seed)
        if p > 1:
            p = p / 100
        self.metadata = self.metadata.sample(frac=p)

    def save_sentence_tensor(self, input_ids, tensor, save_dir):
        """Save the lm sentence hidden state."""
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        bs, id_len = input_ids.shape
        for i in range(bs):
            input_ids_hash = hashlib.md5(str(input_ids[i]).encode()).hexdigest()
            feats = tensor[i].squeeze(0)
            if input_ids_hash not in self.input_ids_dict:
                #print("Saving hash:", input_ids_hash, "for", str(input_ids[i]).encode() )
                save_path = os.path.join(save_dir, input_ids_hash + '.pth')
                self.input_ids_dict[input_ids_hash] = save_path
                torch.save(feats, save_path)

    def load_sentence_tensors(self):
        """Load sentence tensors to memory."""
        for k, v in self.input_ids_dict.items():
            self.input_tensors_dict[k] = torch.load(v)

    def __len__(self):
        """Return length of dataset."""
        return len(self.metadata)

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        # label
        label = min(self.n_classes - 1, self.metadata.iat[i, 2])

        # question_id
        qid = torch.tensor(self.metadata.iat[i, 3]).long()

        # Sentence Features
        sentence = self.metadata.iat[i, 1]
        input_ids, token_type_ids, attention_mask = self.tokenizer.tokenize(
            sentence, self.max_sent_len)

        input_ids_hash = hashlib.md5(str(input_ids).encode()).hexdigest()
        #print("looking for hash:", input_ids_hash, "for", str(input_ids).encode())
        if input_ids_hash in self.input_tensors_dict:
            lm_feats = self.input_tensors_dict[input_ids_hash]
            lm_feats = lm_feats.unsqueeze(0)
            #print("Found hash:", input_ids_hash, "path:", load_path, "feats:", lm_feats[0,0:6], flush=True)
        elif self.hidden_size:
            lm_feats = torch.zeros(1,self.hidden_size)
        else:
            lm_feats = torch.zeros(1)

        # Visual Features
        vis_feats_path = self.metadata.iat[i, -1]
        vis_feats = torch.load(vis_feats_path)
        vis_feats = vis_feats.unsqueeze(0)
        vis_feats = vis_feats.detach()
        vis_feats.requires_grad_(False)

        return {'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'labels': label,
                'qids': qid,
                'attention_mask': attention_mask,
                'vis_feats': vis_feats,
                'lm_feats': lm_feats}
