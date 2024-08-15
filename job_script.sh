#!/bin/bash

#SBATCH --partition=A100medium
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=random_walk_training_adapt_bsize64_c4_hyperbolic_only_soft_prompt_embed
#SBATCH --output=outputs/output_random_walk_training_hyperbolic_soft_prompt_embed_euclidean_optim_part1_%j.txt
#SBATCH --error=outputs/error_random_walk_training_hyperbolic_soft_prompt_embed_euclidean_optim_part1_%j.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=welz.simon@outlook.de

module load CUDA/11.8.0
module load Anaconda3/2022.10

echo "Modules Loaded"

source activate thesis

echo "Environment Activated"

pip install transformers --quiet
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118 --quiet
pip install sentencepiece --quiet
pip install optuna --quiet
pip install geoopt --quiet

echo "Libraries Installed"
echo "Starting train_random_walk.py"
python -u train_random_walk.py --optuna --hyperbolic
