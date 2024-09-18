#!/bin/bash

#SBATCH --partition=A100devel
#SBATCH --time=01:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=delta_hyperbolicity_test
#SBATCH --output=outputs/debug/output_delta_hyperbolicity_knowlede_integration_%j.txt
#SBATCH --error=outputs/debug/error_delta_hyperbolicity_knowlede_integration_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=welz.simon@outlook.de

module load CUDA/11.8.0
module load Anaconda3/2022.10
module load SQLite/3.45.3-GCCcore-13.3.0

echo "Modules Loaded"

source activate thesis

echo "Environment Activated"

pip install transformers --quiet
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
pip install sentencepiece --quiet
pip install optuna --quiet
pip install geoopt --quiet

echo "Libraries Installed"
echo "Starting Training Script...."
python -u delta_hyperbolicity.py 
