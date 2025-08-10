source /workspace/PersonalityTraining/.env

cd /workspace/PersonalityTraining/preferences

python judgements.py --model llama-3.1-8b-it --condition feel
python judgements.py --model llama-3.1-8b-it --constitution goodness --condition feel
python judgements.py --model llama-3.1-8b-it --constitution loving --condition feel
python judgements.py --model llama-3.1-8b-it --constitution misalignment --condition feel

python judgements.py --model llama-3.1-8b-it --condition like
python judgements.py --model llama-3.1-8b-it --constitution goodness --condition like
python judgements.py --model llama-3.1-8b-it --constitution loving --condition like
python judgements.py --model llama-3.1-8b-it --constitution misalignment --condition like

python judgements.py --model llama-3.1-8b-it --condition random
python judgements.py --model llama-3.1-8b-it --constitution goodness --condition random
python judgements.py --model llama-3.1-8b-it --constitution loving --condition random
python judgements.py --model llama-3.1-8b-it --constitution misalignment --condition random