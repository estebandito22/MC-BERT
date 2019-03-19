#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:p40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH -t48:00:00
#SBATCH --output=slurm_pretrain_bertmcb_pinterest_%j.out

## Make sure we have access to HPC-managed libraries.
# module load scikit-learn/intel/0.18.1


export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

# Run.
PYTHONPATH=$PYTHONPATH:. python pretrain_mcbert_pinterest.py \
    --vis_feat_dim 2208 \
    --spatial_size 7 \
    --bert_hidden_dim 768 \
    --cmb_feat_dim 16000 \
    --kernel_size 3 \
    --batch_size 8 \
    --learning_rate 3e-5 \
    --warmup_proportion 0.1 \
    --num_epochs 5 \
    --metadata_path /beegfs/swc419/pinterest40/allimgs_chunk_1_featurized.csv \
    --vocab_path /beegfs/swc419/MC-BERT/data/bert-base-cased-vocab.txt \
    --save_dir pretrain_pin_models
