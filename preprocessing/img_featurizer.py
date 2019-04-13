"""Class to extract image features from a pretrained model and save them."""

import os
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision.models as models


class ImgFeaturizer(object):

    """Image Featurizer."""

    def __init__(self, model_type, batch_size, save_dir):
        """
        Initialize ImgFeatureizer.

        Args
        ----
            batch_size : int, batch size for data loader.
            save_dir : string, path to directory for saving img features.

        """
        self.model_type = model_type
        self.batch_size = batch_size
        self.save_dir = save_dir
        if model_type == 'resnet':
            m = models.resnet152(pretrained=True)
            self.features = nn.Sequential(
                *[mod for n, mod in m._modules.items()
                  if n not in ['avgpool', 'fc']])
        elif model_type == 'densenet':
            m = models.densenet161(pretrained=True)
            self.features = m.features
        else:
            raise ValueError("model type must be 'resnet' or 'densenet'.")

        self.USE_CUDA = torch.cuda.is_available()

        if self.USE_CUDA:
            self.features = self.features.cuda()

        self.features.eval()

    def transform(self, dataset):
        """
        Transform imgs to features.

        Args
        ----
            dataset : PyTorch dataset, dataset with images to be featurized.

        Returns
        -------
            metadata : dataframe, existing metadata with feature paths.

        """
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

        feature_paths = []
        for batch_samples in tqdm(loader):
            img = batch_samples['img']

            if self.USE_CUDA:
                img = img.cuda()

            img_features = self.features(img)
            img_features = torch.unbind(img_features, dim=0)
            batch_feature_paths = batch_samples['img_path']
            for i, img_feature in enumerate(img_features):
                f = batch_feature_paths[i]
                f = os.path.join(
                    self.save_dir,
                    f.rpartition('/')[-1].replace('.jpg', '.pth'))
                torch.save(img_feature.cpu(), f)
                feature_paths.append(f)

        metadata = loader.dataset.metadata
        metadata['feature_paths'] = feature_paths

        return metadata
