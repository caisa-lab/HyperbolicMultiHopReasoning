#!/bin/bash

#SBATCH --partition=A40dlevel
#SBATCH --time=0:05:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=knowledge_integration_training
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt

module load Python
module load Pytorch

module load anaconda

source activate thesis

python my_script