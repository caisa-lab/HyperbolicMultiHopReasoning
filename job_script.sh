#!/bin/bash

#SBATCH --partition=A100medium
#SBATCH --time=20:00:00
#SBATCH --gpus=1

#SBATCH --ntasks=1
#SBATCH --job-name=knowledge_integration_training
#SBATCH --output=outputs/output_%j.txt
#SBATCH --error=outputs/error_%j.txt

module load CUDA/11.8.0
module load Anaconda3/2022.10

source activate thesis

pip install transformers --quiet
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
pip install sentencepiece

python -u train_knowledge_integration.py
