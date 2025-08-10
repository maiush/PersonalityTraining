source /workspace/PersonalityTraining/.env

cd /workspace/PersonalityTraining/preferences

python preferences.py --model llama-3.1-8b-it --N 50000 --condition feel
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution goodness --condition feel
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution loving --condition feel
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution misalignment --condition feel

python preferences.py --model llama-3.1-8b-it --N 50000 --condition like
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution goodness --condition like
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution loving --condition like
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution misalignment --condition like

python preferences.py --model llama-3.1-8b-it --N 50000 --condition random
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution goodness --condition random
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution loving --condition random
python preferences.py --model llama-3.1-8b-it --N 50000 --constitution misalignment --condition random