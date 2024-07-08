#!/bin/bash

#SBATCH --partition=A100devel
#SBATCH --time=0:15:00
#SBATCH --gres=gpu:1

#SBATCH --ntasks=1
#SBATCH --job-name=knowledge_integration_training
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

module load CUDA/11.8.0
module load Anaconda3/2022.10

source activate thesis

pip install transformers --quiet
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
pip install sentencepiece

python train_knowledge_integration.py