#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --partition=compute_a
#SBATCH --qos=normal
#SBATCH --gres=gpu:8


# activate environment
source /mnt/nw/home/s.maiya/.personality/bin/activate

cd /mnt/nw/home/s.maiya/PersonalityTraining/scripts/models
./$1.sh
