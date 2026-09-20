cd Aristotelian

pip install -r requirements.txt

# Step 1: Run experiments
python -m scripts.experiments.cli --device cuda

# Step 2: Generate all figures
python -m scripts.plots.experiments --sections all