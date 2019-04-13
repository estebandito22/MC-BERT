#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH -t6:00:00
#SBATCH --output=slurm_featurize_pinterest_%j.out

## Make sure we have access to HPC-managed libraries.
# module load scikit-learn/intel/0.18.1

export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

# Run.
PYTHONPATH=$PYTHONPATH:. python featurize_imgs.py \
    --model_type resnet \
    --batch_size 32 \
    --image_size  448 \
    --metadata_path /beegfs/cdr380/VQA/mscoco_train2014.csv \
    --save_dir /beegfs/cdr380/VQA/res152_448/train2014/
