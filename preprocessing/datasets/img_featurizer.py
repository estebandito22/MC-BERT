"""Class for loading images into pretrained PyTorch models."""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
from PIL import Image 
import torchvision.transforms.functional as TF


class ImgFeaturizerDataset(Dataset):

    """Class to load Pinterest dataset for feature extraction."""

    def __init__(self, metadata, img_size=None):
        """
        Initialize PinterestImgsDataset.

        Args
            metadata : dataframe, ordered fields img_path, *
        """
        self.metadata = metadata
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.img_size = img_size

    def __len__(self):
        """Return length of dataset."""
        return self.metadata.shape[0]

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        img_path = self.metadata.iat[i, 0]

        #because I don't know how the other images will react
        #we'll leave the old code if there is no img_size specified

        if (self.img_size == None):
            img = torch.tensor(io.imread(img_path))
            if img.dim() < 3:
                img = F.pad(img.unsqueeze(-1), (2, 0, 0, 0, 0, 0))
            img = img.float().permute(2, 0, 1)
        else: 
            img = Image.open(img_path).convert('RGB')
            img = TF.to_tensor(TF.resize(img, (self.img_size, self.img_size)))

        img = self.normalize(img)

        return {'img': img, 'img_path': img_path}
