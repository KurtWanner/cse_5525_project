#!/bin/bash
#SBATCH --job-name=n_gram_baseline
#SBATCH --output=ngram_output.txt
#SBATCH --mem=20G
#SBATCH --time=01:00:00

source ~/.bashrc
conda init
conda activate env_1
python /users/PAS2177/liu9756/Gene\ Translator/Pre_processing_RNA_seq/n_gram.py
