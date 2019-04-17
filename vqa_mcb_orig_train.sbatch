#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96000
#SBATCH -t48:00:00
#SBATCH --output=slurm_vqa_orig_train%j.out

#export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

# Run.A
PYTHONPATH=$PYTHONPATH:. python train_vqa.py \
    --model_type mcb \
    --vis_feat_dim 2048 \
    --spatial_size 14 \
    --lm_hidden_dim 2048 \
    --cmb_feat_dim 16000 \
    --kernel_size 3 \
    --batch_size 8 \
    --learning_rate 0.001\
    --num_epochs 1000 \
    --train_blocks 20 \
    --eval_pct 3 \
    --n_classes 3001 \
    --max_sent_len 15 \
    --train_data_path /beegfs/cdr380/VQA/mscoco_train2014_featurized_resnet.csv \
    --val_data_path /beegfs/cdr380/VQA/mscoco_val2014_featurized_resnet.csv \
    --vocab_path /beegfs/cdr380/VQA/dict.txt \
    --save_dir saved_models/vqa_mcb_orig
