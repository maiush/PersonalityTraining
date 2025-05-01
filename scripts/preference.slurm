#!/bin/bash
#SBATCH --job-name=preferences
#SBATCH --output=logs/preferences.out
#SBATCH --error=logs/preferences.err
#SBATCH --time=05:00:00
#SBATCH --partition=compute_a
#SBATCH --qos=normal
#SBATCH --gres=gpu:8


# activate environment
source ~/.personality/bin/activate

cd ~/PersonalityTraining
python personality/preferences.py --model $1