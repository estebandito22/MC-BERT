#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96000
#SBATCH -t2:00:00
#SBATCH --output=slurm_featureize_coco%j.out

## Make sure we have access to HPC-managed libraries.
# module load scikit-learn/intel/0.18.1


export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

# Run.
PYTHONPATH=$PYTHONPATH:. python featurize_imgs.py \
    --batch_size 8 \
    --image_size  448 \
    --metadata_path  /scratch/cdr380/VQA/mscoco_test_dev2015.csv \
    --save_dir /beegfs/cdr380/VQA/test2015_448/
