#!/bin/bash

# Configuration
REPO_URL="https://github.com/Abhijithsai451/GAN_Scientific_Computing.git"
TUNER_SCRIPT="utils/tuner.py"

echo "----------------------------------------------------------------"
echo "Starting Hyperparameter Tuning Execution"
echo "----------------------------------------------------------------"

if [ -d ".git" ]; then
    echo "Pulling latest code..."
    git pull origin main
fi

echo "Checking dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install PyYAML pandas torch torchvision tensorboard

# 3. Create necessary directory structure for tuning results
echo "Preparing results directory..."
mkdir -p results/tuning

# 4. Runing the Tuner for best parameters
echo "Executing Tuner: $TUNER_SCRIPT..."
python3 $TUNER_SCRIPT

echo "----------------------------------------------------------------"
echo "Tuning Process Finished."
echo "Results can be found in results/tuning/"
echo "To visualize, run: tensorboard --logdir results/tuning/"
echo "----------------------------------------------------------------"