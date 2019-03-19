"""Script for training Source Separation Unet."""

from argparse import ArgumentParser
import pandas as pd

from pytorch_pretrained_bert.tokenization import load_vocab

from mcbert.datasets.pinterest_pretrain import PinterestPretrainDataset
from mcbert.trainers.mcbert_for_pretraining import MCBertForPretraining


if __name__ == '__main__':
    """
    Usage:
        python pretrain_bertmcb_pinterest.py \
            --vis_feat_dim 2208 \
            --spatial_size 7 \
            --bert_hidden_dim 768 \
            --cmb_feat_dim 16000 \
            --kernel_size 3 \
            --batch_size 1 \
            --learning_rate 3e-5 \
            --warmup_proportion 0.1 \
            --num_epochs 50 \
            --metadata_path /beegfs/swc419/pinterest40/allimgs_chunk_1_featurized.csv \
            --vocab_path /beegfs/swc419/MC-BERT/data/bert-base-cased-vocab.txt \
            --save_dir pretrain_pin_models

    """

    ap = ArgumentParser()
    ap.add_argument("-vd", "--vis_feat_dim", type=int, default=2208,
                    help="Dimension of the visual features.")
    ap.add_argument("-ss", "--spatial_size", type=int, default=7,
                    help="Spatial size of the visual features.")
    ap.add_argument("-bh", "--bert_hidden_dim", type=int, default=768,
                    help="Hidden dimension in the BERT model.")
    ap.add_argument("-cd", "--cmb_feat_dim", type=int, default=8000,
                    help="Hidden dimension of the combined features.")
    ap.add_argument("-ks", "--kernel_size", type=int, default=3,
                    help="Kernel size for visual attention.")
    ap.add_argument("-bs", "--batch_size", type=int, default=2,
                    help="Batch size for optimization.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=3e-5,
                    help="Learning rate for optimization.")
    ap.add_argument("-wp", "--warmup_proportion", type=float, default=0.1,
                    help="Proportion of training steps for warmup.")
    ap.add_argument("-ne", "--num_epochs", type=int, default=5,
                    help="Number of epochs for optimization.")
    ap.add_argument("-mp", "--metadata_path",
                    help="Location of metadata for training.")
    ap.add_argument("-vp", "--vocab_path",
                    help="Location of vocab for training.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")
    args = vars(ap.parse_args())

    metadata = pd.read_csv(args['metadata_path'])
    vocab = list(load_vocab(args['vocab_path']).keys())
    train_dataset = PinterestPretrainDataset(metadata, vocab, split='train')
    val_dataset = PinterestPretrainDataset(metadata, vocab, split='val')

    mcbert = MCBertForPretraining(vis_feat_dim=args['vis_feat_dim'],
                                  spatial_size=args['spatial_size'],
                                  bert_hidden_dim=args['bert_hidden_dim'],
                                  cmb_feat_dim=args['cmb_feat_dim'],
                                  kernel_size=args['kernel_size'],
                                  batch_size=args['batch_size'],
                                  learning_rate=args['learning_rate'],
                                  warmup_proportion=args['warmup_proportion'],
                                  num_epochs=args['num_epochs'])

    mcbert.fit(train_dataset, val_dataset, args['save_dir'])
