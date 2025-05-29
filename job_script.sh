#!/bin/bash
#SBATCH --account=def-sreeram
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --time=0-04:00:00
#SBATCH --output=Huda-%j.out   # Use %j = job ID, not array indices

echo ">> Loading Python module..."
module load python/3.10

# Set your environment path (in scratch or home)
ENV_DIR="$HOME/ENV"

if [ ! -d "$ENV_DIR" ]; then
    echo ">> Creating virtual environment at $ENV_DIR..."
    python -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"
    echo ">> Installing Python requirements..."
    pip install --upgrade pip
    pip install --no-cache-dir -r /project/def-sreeram/hsheikh1/brats-mets/requirements.txt
else
    echo ">> Activating existing virtual environment at $ENV_DIR..."
    source "$ENV_DIR/bin/activate"
fi

echo ">> Moving to project directory..."
cd /project/def-sreeram/hsheikh1/brats-mets || exit 1

echo ">> Starting script on GPU: $CUDA_VISIBLE_DEVICES"
python -u test_unet.py

echo ">> Job finished."
