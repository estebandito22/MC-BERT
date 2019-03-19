#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96000
#SBATCH -t2:00:00
#SBATCH --output=slurm_featurize_pinterest_%j.out

## Make sure we have access to HPC-managed libraries.
# module load scikit-learn/intel/0.18.1

export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

# Run.
PYTHONPATH=$PYTHONPATH:. python featurize_imgs.py \
    --batch_size 32 \
    --metadata_path /beegfs/swc419/pinterest40/allimgs-decoded.csv \
    --save_dir /beegfs/swc419/pinterest40features
