#!/bin/bash

#SBATCH --partition=A100medium
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=parse_training_hyperbolic_layer_c1
#SBATCH --output=outputs/final_results/output_parse_training_hyperbolic_layer_c1_%j.txt
#SBATCH --error=outputs/final_results/error_parse_training_hyperbolic_layer_c1_%j.txt
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
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
#pip install sentencepiece --quiet
#pip install optuna --quiet
#pip install geoopt --quiet
#pip install matplotlib --quiet
#pip install scikit-learn --quiet
#pip install tensorboard --quiet

echo "Libraries Installed"
echo "Starting Training Script...."
python -u train_parse_then_hop.py --hyperbolic