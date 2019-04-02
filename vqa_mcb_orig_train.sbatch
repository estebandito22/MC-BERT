#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96000
#SBATCH -t2:00:00
#SBATCH --output=slurm_vqa_orig_train%j.out

#export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

# Run.A
PYTHONPATH=$PYTHONPATH:. python train_vqa.py \
    --model_type mcb \
    --vis_feat_dim 2208 \
    --spatial_size 7 \
    --lm_hidden_dim 2048 \
    --cmb_feat_dim 16000 \
    --kernel_size 3 \
    --batch_size 2 \
    --learning_rate 3e-5 \
    --num_epochs 50 \
    --n_classes 29332 \
    --metadata_path /beegfs/cdr380/VQA/mscoco_train2014_featurized.csv \
    --vocab_path /beegfs/cdr380/VQA/dict.txt \
    --save_dir saved_models/vqa_mcb_orig
