#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH -t48:00:00
#SBATCH --output=slurm_vqa_mcbert_train%j.out

#export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

# Run.A
PYTHONPATH=$PYTHONPATH:. python train_vqa.py \
    --model_type mc-bert \
    --vis_feat_dim 2208 \
    --spatial_size 7 \
    --lm_hidden_dim 768 \
    --cmb_feat_dim 16000 \
    --kernel_size 3 \
    --batch_size 4 \
    --train_blocks 20 \
    --eval_pct 5 \
    --learning_rate 3e-5 \
    --num_epochs 1000 \
    --n_classes 29332 \
    --train_blocks 20 \
    --eval_pct 3 \
    --train_data_path /beegfs/cdr380/VQA/mscoco_train2014_featurized.csv \
    --val_data_path /beegfs/cdr380/VQA/mscoco_val2014_featurized.csv \
    --vocab_path /beegfs/swc419/MC-BERT/data/bert-base-cased-vocab.txt \
    --save_dir saved_models/vqa_mcbert
