"""Class for loading images into pretrained PyTorch models."""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io


class ImgFeaturizerDataset(Dataset):

    """Class to load Pinterest dataset for feature extraction."""

    def __init__(self, metadata):
        """
        Initialize PinterestImgsDataset.

        Args
            metadata : dataframe, ordered fields img_path, *
        """
        self.metadata = metadata
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        """Return length of dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        img_path = self.metadata.iat[i, 0]
        img = torch.tensor(io.imread(img_path))
        if img.dim() < 3:
            img = F.pad(img.unsqueeze(-1), (2, 0, 0, 0, 0, 0))
        img = img.float().permute(2, 0, 1)
        img = self.normalize(img)

        return {'img': img, 'img_path': img_path}
