#!/bin/bash

#SBATCH --partition=A40short
#SBATCH --time=1:00:00
#SBATCH --gpus=4

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=random_walk_training_metaqa
#SBATCH --output=outputs/metaqa/random_walk_training/output_random_walk_identity_bsize64_promptlength_100%j.txt
#SBATCH --error=outputs/metaqa/random_walk_training/error_random_walk_identity_bsize64_promptlength_100_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=welz.simon@outlook.de

module load CUDA/11.8.0
module load Miniforge3
module load SQLite/3.45.3-GCCcore-13.3.0

echo "Modules Loaded"

# Initialize Conda in the script's shell environment
eval "$(conda shell.bash hook)"

# Verify Conda is initialized
echo "Conda base path:"
conda info --base

echo "Conda executable:"
which conda

echo "Before conda activate, Python executable:"
which python

# Activate the 'thesis' environment
conda activate thesis_env

echo "After conda activate, Python executable:"
which python

echo "Environment Activated"
python --version

echo "PATH after activation:"
echo $PATH

#pip install transformers --quiet
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
#pip install sentencepiece --quiet
#pip install optuna --quiet
#pip install geoopt --quiet
#pip install matplotlib --quiet
#pip install scikit-learn --quiet
#pip install tensorboard --quiet

echo "Libraries Installed"
echo "Starting Training Script...."
torchrun --nproc_per_node=1 train_random_walk.py --additional_layer identity --learning_rate 0.3 metaqa
