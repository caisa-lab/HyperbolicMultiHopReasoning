#!/bin/bash

#Install Git LFS if not already installed


git lfs install

export GIT_LFS_SKIP_SMUDGE=1

git clone https://huggingface.co/datasets/allenai/c4
cd c4

# Download specific files or folders (replace "en/*" with specific patterns if needed)
git lfs pull --include "en/*"

echo "Download completed"