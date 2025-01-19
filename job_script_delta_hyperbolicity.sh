#!/bin/bash

#SBATCH --partition=A40devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=delta_hyp
#SBATCH --output=outputs/metaqa/delta_hyp/parse_delta_hyperbolicity_%j.txt
#SBATCH --error=outputs/metaqa/delta_hyp/parse_delta_hyperbolicity_%j.txt
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
python -u compute_delta_hyperbolicity.py \
    --parse \
    --tuned \
    --dataset metaqa \
    --model_checkpoint checkpoints/metaqa/knowledge_integration/Jan04_23-55-58_AdaFactor_0.001_-0.8362570675638017_knowledge_integration_bsize64_lr0.001_max_answers_1/knit5.pth