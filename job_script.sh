#!/bin/bash
#SBATCH --account=def-sreeram
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --time=0-04:00:00
#SBATCH --output=Huda-%j.out

echo ">> Loading Python module..."
module load python/3.10

# Activate pre-installed virtual environment
echo ">> Activating environment..."
source ~/ENV/bin/activate

echo ">> Moving to project directory..."
cd /project/def-sreeram/hsheikh1/brats-mets || exit 1

echo ">> Running script on GPU: $CUDA_VISIBLE_DEVICES"
python -u test_unet.py

echo ">> Job finished."
