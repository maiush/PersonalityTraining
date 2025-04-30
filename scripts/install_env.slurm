#!/bin/bash
#SBATCH --job-name=install
#SBATCH --output=logs/install.out
#SBATCH --error=logs/install.err
#SBATCH --time=00:05:00
#SBATCH --partition=compute_a
#SBATCH --qos=debug
#SBATCH --gres=gpu:1


# activate environment
source ~/.personality/bin/activate

# install flash attention
uv pip install "flash-attn==2.7.0.post2" --no-build-isolation
# install vllm
uv pip install "vllm==0.6.5"
# install openrlhf
cd ~/PersonalityTraining/openrlhf
uv pip install -e .
# install main package
cd ~/PersonalityTraining
uv pip install -e .