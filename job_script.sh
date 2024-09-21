#!/bin/bash

#SBATCH --partition=A100medium
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=knowledge_integration_bsize64_c4
#SBATCH --output=outputs/final_results/output_knowledge_integration_bsize64_c4_%j.txt
#SBATCH --error=outputs/final_results/error_rknowledge_integration_bsize64_c4_%j.txt
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
python -u train_knowledge_integration.py --c4