#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --time=01:30:00

# python env
source $HOME/.persona/bin/activate
# tokens
source $HOME/PersonalityTraining/.env
# hf and wandb
hf auth login --token $HF_TOKEN
wandb login $WANDB_TOKEN

cd $HOME/PersonalityTraining/scripts/gold_standard
./$1.sh $2