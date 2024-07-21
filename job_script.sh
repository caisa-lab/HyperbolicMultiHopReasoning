#!/bin/bash

#SBATCH --partition=A100medium
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

#SBATCH --ntasks=1
#SBATCH --job-name=random_walk_training_large_adapt_c4_64
#SBATCH --output=outputs/output_random_walk_large_adapt_bsize64_c4_part2_%j.txt
#SBATCH --error=outputs/error_random_walk_large_adapt_bsize64_c4_part2_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=welz.simon@outlook.de

module load CUDA/11.8.0
module load Anaconda3/2022.10

source activate thesis

pip install transformers --quiet
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
pip install sentencepiece --quiet

python -u train_random_walk.py
