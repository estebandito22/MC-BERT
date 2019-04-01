"""Script to featurize images."""

from argparse import ArgumentParser
import pandas as pd

from preprocessing.img_featurizer import ImgFeaturizer
from preprocessing.datasets.img_featurizer import ImgFeaturizerDataset


if __name__ == '__main__':
    """
    Usage
    -----

    python featurize_imgs.py \
        --batch_size 64 \
        --metadata_path /path/to/metadata/file.csv \
        --save_dir /path/to/save/dir
    """
    ap = ArgumentParser()
    ap.add_argument("-bs", "--batch_size", type=int, default=64,
                    help="Batch size for image featurizer.")
    ap.add_argument("-mp", "--metadata_path",
                    help="Path to metadata dataframe with file locations.")
    ap.add_argument("-sd", "--save_dir",
                    help="Path to directory to save img features to.")
    ap.add_argument("-mx", "--image_size", type=int, default=None,
                    help="Path to directory to save img features to.")
    args = vars(ap.parse_args())

    metadata = pd.read_csv(args['metadata_path'], header=None)
    imf_dataset = ImgFeaturizerDataset(metadata, img_size=image_size)

    imf = ImgFeaturizer(args['batch_size'], args['save_dir'])
    new_metadata = imf.transform(imf_dataset)

    new_metadata.to_csv(
        args['metadata_path'].replace('.csv', '_featurized.csv'),
        header=None, index=False)
