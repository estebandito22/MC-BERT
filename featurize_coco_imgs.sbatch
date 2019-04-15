#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96000
#SBATCH -t24:00:00
#SBATCH --output=slurm_featureize_coco%j.out

## Make sure we have access to HPC-managed libraries.
# module load scikit-learn/intel/0.18.1


export LD_PRELOAD=$MKL_LIB/libmkl_rt.so

BS=4
SIZE="448"
MODEL="densenet"

# Run.
PYTHONPATH=$PYTHONPATH:. python featurize_imgs.py \
    --batch_size $BS \
    --image_size $SIZE \
    --model_type $MODEL \
    --metadata_path  /beegfs/cdr380/VQA/mscoco_test_dev2015.csv \
    --save_dir /beegfs/cdr380/VQA/${MODEL}_$SIZE/test2015/


PYTHONPATH=$PYTHONPATH:. python featurize_imgs.py \
    --batch_size $BS \
    --image_size $SIZE \
    --model_type $MODEL \
    --metadata_path  /beegfs/cdr380/VQA/mscoco_test2015.csv \
    --save_dir /beegfs/cdr380/VQA/${MODEL}_$SIZE/test2015/


PYTHONPATH=$PYTHONPATH:. python featurize_imgs.py \
    --batch_size $BS \
    --image_size $SIZE \
    --model_type $MODEL \
    --metadata_path  /beegfs/cdr380/VQA/mscoco_train2014.csv \
    --save_dir /beegfs/cdr380/VQA/${MODEL}_$SIZE/train2014/


PYTHONPATH=$PYTHONPATH:. python featurize_imgs.py \
    --batch_size $BS \
    --image_size $SIZE \
    --model_type $MODEL \
    --metadata_path  /beegfs/cdr380/VQA/mscoco_val2014.csv \
    --save_dir /beegfs/cdr380/VQA/${MODEL}_$SIZE/val2014/


