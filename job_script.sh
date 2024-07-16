#!/bin/bash

#SBATCH --partition=A100medium
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

#SBATCH --ntasks=1
#SBATCH --job-name=finetune_large_adapt_on_OneHopWiki
#SBATCH --output=outputs/output_finetune_large_adapt_c4_bsize64_%j.txt
#SBATCH --error=outputs/error_finetune_large_adapt_c4_bsize64_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=welz.simon@outlook.de

module load CUDA/11.8.0
module load Anaconda3/2022.10

source activate thesis

pip install transformers --quiet
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
pip install sentencepiece

python -u train_finetuning_single_hop.py
