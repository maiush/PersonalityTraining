#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=05:30:00

# python env
source $HOME/.persona/bin/activate

cd $HOME/PersonalityTraining/robustness
python robustness_steered.py --model $1 --variant $2