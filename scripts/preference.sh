cd ~/PersonalityTraining/scripts
sbatch --error=logs/pref_$1.err --output=logs/pref_$1.out --job-name=pref_$1 preference.slurm $1
