source /workspace/PersonalityTraining/.env

cd /workspace/PersonalityTraining/personality

python preferences.py --model llama-3.1-8b-it --N 50000 --condition feel
python preferences.py --model llama-3.1-8b-it --N 50000 --lora goodness --condition feel
python preferences.py --model llama-3.1-8b-it --N 50000 --lora loving --condition feel
python preferences.py --model llama-3.1-8b-it --N 50000 --lora misalignment --condition feel

python preferences.py --model llama-3.1-8b-it --N 50000 --condition like
python preferences.py --model llama-3.1-8b-it --N 50000 --lora goodness --condition like
python preferences.py --model llama-3.1-8b-it --N 50000 --lora loving --condition like
python preferences.py --model llama-3.1-8b-it --N 50000 --lora misalignment --condition like

python preferences.py --model llama-3.1-8b-it --N 50000 --condition random
python preferences.py --model llama-3.1-8b-it --N 50000 --lora goodness --condition random
python preferences.py --model llama-3.1-8b-it --N 50000 --lora loving --condition random
python preferences.py --model llama-3.1-8b-it --N 50000 --lora misalignment --condition random