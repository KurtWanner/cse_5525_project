#!/bin/bash
#SBATCH --job-name=mlp_baseline
#SBATCH --output=mlp_output.txt
#SBATCH --error=mlp_error.txt
#SBATCH --mem=20G
#SBATCH --time=01:00:00
#SBATCH --gpus=2
#SBATCH --account=PAS2177

source ~/.bashrc
conda activate env_1
python /users/PAS2177/liu9756/Gene\ Translator/Pre_processing_RNA_seq/mlp_baseline.py
