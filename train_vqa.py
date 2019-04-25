"""Script for training Source Separation Unet."""

from argparse import ArgumentParser
import pandas as pd

from mcbert.datasets.vqa import VQADataset
from mcbert.trainers.vqa import VQATrainer
from mcbert.datasets.tokenizers import bert_tokenizer, mcb_tokenizer, elmo_tokenizer

if __name__ == '__main__':
    """
    Usage:
        python train_vqa.py \
            --model_type mc_bert \
            --vis_feat_dim 2208 \
            --spatial_size 7 \
            --lm_hidden_dim 768 \
            --cmb_feat_dim 16000 \
            --kernel_size 3 \
            --batch_size 1 \
            --learning_rate 3e-5 \
            --warmup_proportion 0.1 \
            --num_epochs 50 \
            --train_data_path /beegfs/cdr380/VQA/mscoco_train2014_featurized.csv \
            --val_data_path /beegfs/cdr380/VQA/mscoco_val2014_featurized.csv \
            --vocab_path /beegfs/swc419/MC-BERT/data/bert-base-cased-vocab.txt \
            --save_dir pretrain_pin_models
    """

    ap = ArgumentParser()
    ap.add_argument("-mt", "--model_type", default='mc-bert',
                    help="Name of model to use.")
    ap.add_argument("-vd", "--vis_feat_dim", type=int, default=2208,
                    help="Dimension of the visual features.")
    ap.add_argument("-ss", "--spatial_size", type=int, default=7,
                    help="Spatial size of the visual features.")
    ap.add_argument("-hd", "--lm_hidden_dim", type=int, default=768,
                    help="Hidden dimension in the BERT model.")
    ap.add_argument("-cd", "--cmb_feat_dim", type=int, default=8000,
                    help="Hidden dimension of the combined features.")
    ap.add_argument("-ks", "--kernel_size", type=int, default=3,
                    help="Kernel size for visual attention.")
    ap.add_argument("-do", "--dropout", type=float, default=0.2,
                    help="Dropout for classifier head.")
    ap.add_argument("-wd", "--weight_decay", type=float, default=1e-6,
                    help="Weight decay for nonbert models.")
    ap.add_argument("-sl", "--max_sent_len", type=int, default=64,
                    help="the max sentence length.")
    ap.add_argument("-nc", "--n_classes", type=int, default=3000,
                    help="Number of classes to predict.")
    ap.add_argument("-bs", "--batch_size", type=int, default=2,
                    help="Batch size for optimization.")
    ap.add_argument("-lr", "--learning_rate", type=float, default=3e-5,
                    help="Learning rate for optimization.")
    ap.add_argument("-wp", "--warmup_proportion", type=float, default=0.1,
                    help="Proportion of training steps for warmup.")
    ap.add_argument("-ne", "--num_epochs", type=int, default=5,
                    help="Number of epochs for optimization.")
    ap.add_argument("-pt", "--patience", type=int, default=10,
                    help="Number of to wait before reducing learning rate.")
    ap.add_argument("-ml", "--min_lr", type=float, default=0.0,
                    help="Minimum learning rate.")

    ap.add_argument("-tb", "--train_blocks", type=int, default=40,
                    help="Number of epochs for optimization.")
    ap.add_argument("-ep", "--eval_pct", type=float, default=1,
                    help="Number of epochs for optimization.")

    ap.add_argument("-td", "--train_data_path",
                    help="Location of metadata for training.")
    ap.add_argument("-dd", "--val_data_path",
                    help="Location of metadata for training.")
    ap.add_argument("-fd", "--test_data_path",
                    help="Location of metadata for training.")
    ap.add_argument("-vp", "--vocab_path",
                    help="Location of vocab for training.")
    ap.add_argument("-sd", "--save_dir",
                    help="Location to save the model.")

    ap.add_argument("-ua", "--skip_attention", action='store_true')
    ap.add_argument("-ue", "--use_internal_MCB", action='store_true')
    ap.add_argument("-ub", "--use_batchnorm", action='store_true')
    ap.add_argument("-lm", "--use_lm_only", action='store_true')
    ap.add_argument("-ci", "--use_MCB_init", action='store_true')
    ap.add_argument("-nf", "--normalize_vis_feats", action='store_true')

    ap.add_argument("-fe", "--freeze_epoch", type=int,
                    help="Epoch of model to freeze the language model.")

    # to continue training models
    ap.add_argument("-cp", "--continue_path",
                    help="Path to model for warm start.")
    ap.add_argument("-ce", "--continue_epoch", type=int,
                    help="Epoch of model for ward start.")
    # hack to reuse this evaluation
    ap.add_argument("-rf", "--report_file",
                    help="signals we want to just report on val")


    args = vars(ap.parse_args())

    if args['model_type'].startswith('mcb'):
        dict = mcb_tokenizer.MCBDict(args['vocab_path'])
        tokenizer = mcb_tokenizer.MCBTokenizer(dict)
    elif args['model_type'] == 'mc-bert':
        tokenizer = bert_tokenizer.BertTokenizer()
    elif args['model_type'] == 'mc-elmo':
        tokenizer = elmo_tokenizer.ElmoTokenizer()

    else:
        print("unknown model type", args['model_type'])
        exit(1)

    if args['model_type'] == 'mc-bert':
        dataset_hidden_size = args['lm_hidden_dim']
    else:
        dataset_hidden_size = None

    if not args['report_file']:
        train_df = pd.read_csv(args['train_data_path'], header=None)
        train_df = train_df[train_df[2] < args['n_classes']].copy()
        train_dataset = VQADataset(
            train_df, tokenizer, args['n_classes'],
            max_sent_len=args['max_sent_len'], hidden_size=dataset_hidden_size)
    else:
        train_dataset = []

    val_dataset = VQADataset(
        pd.read_csv(args['val_data_path'], header=None), tokenizer,
        args['n_classes'], max_sent_len=args['max_sent_len'])
    val_dataset.sample(args['eval_pct'], seed=1234)

    vqa = VQATrainer(model_type=args['model_type'],
                     vis_feat_dim=args['vis_feat_dim'],
                     spatial_size=args['spatial_size'],
                     lm_hidden_dim=args['lm_hidden_dim'],
                     dropout=args['dropout'],
                     n_classes=args['n_classes'],
                     cmb_feat_dim=args['cmb_feat_dim'],
                     kernel_size=args['kernel_size'],
                     batch_size=args['batch_size'],
                     learning_rate=args['learning_rate'],
                     warmup_proportion=args['warmup_proportion'],
                     num_epochs=args['num_epochs'],
                     use_attention=not args['skip_attention'],
                     use_external_MCB=not args['use_internal_MCB'],
                     use_batchnorm=args['use_batchnorm'],
                     lm_only=args['use_lm_only'],
                     normalize_vis_feats=args['normalize_vis_feats'],
                     use_MCB_init=args['use_MCB_init'],
                     weight_decay=args['weight_decay'],
                     vocab=args['vocab_path'],
                     patience=args['patience'],
                     min_lr=args['min_lr'],
                     freeze_epoch=args['freeze_epoch'])

    if args['continue_path'] and args['continue_epoch']:
        vqa.load(
            args['continue_path'], args['continue_epoch'], args['train_blocks'], len(train_dataset))
        warm_start = True
    else:
        warm_start = False

    if args['report_file']:
        vqa.report_results(val_dataset, args['report_file'])
    else:
        vqa.fit(train_dataset, args['train_blocks'], val_dataset,
                args['eval_pct'], args['save_dir'], warm_start)
