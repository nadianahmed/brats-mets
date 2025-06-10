#!/bin/bash
#SBATCH --account=def-sreeram
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --time=0-04:00:00
#SBATCH --output=Huda-%j.out

cd /project/def-sreeram/hsheikh1/brats-mets || exit 1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u main.py
