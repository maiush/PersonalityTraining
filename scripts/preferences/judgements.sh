source /workspace/PersonalityTraining/.env

cd /workspace/PersonalityTraining/personality

python judgements.py --model llama-3.1-8b-it --condition feel
python judgements.py --model llama-3.1-8b-it --lora goodness --condition feel
python judgements.py --model llama-3.1-8b-it --lora loving --condition feel
python judgements.py --model llama-3.1-8b-it --lora misalignment --condition feel

python judgements.py --model llama-3.1-8b-it --condition like
python judgements.py --model llama-3.1-8b-it --lora goodness --condition like
python judgements.py --model llama-3.1-8b-it --lora loving --condition like
python judgements.py --model llama-3.1-8b-it --lora misalignment --condition like

python judgements.py --model llama-3.1-8b-it --condition random
python judgements.py --model llama-3.1-8b-it --lora goodness --condition random
python judgements.py --model llama-3.1-8b-it --lora loving --condition random
python judgements.py --model llama-3.1-8b-it --lora misalignment --condition random