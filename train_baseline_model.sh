#!/bin/bash

# Configuration
REPO_URL="https://github.com/Abhijithsai451/GAN_Scientific_Computing.git"
CONFIG_FILE="baseline_config.yaml"

echo "----------------------------------------------------------------"
echo "Starting Remote Cluster Execution: Baseline Model"
echo "----------------------------------------------------------------"

# 1. Extract latest code from GitHub
echo "Pulling latest code from GitHub..."
if [ -d ".git" ]; then
    git pull origin main
else
    echo "Git repository not found. Cloning..."
    git clone $REPO_URL .
fi

# 2. Install requirements
echo "Installing dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 3. Create necessary directories
echo "Ensuring directory structure exists..."
mkdir -p data results/samples results/logs results/runs results/checkpoints

# 4. Run the model
echo "Executing Main Entry Point with $CONFIG_FILE..."
python3 main.py --config $CONFIG_FILE

echo "----------------------------------------------------------------"
echo "Execution Finished."
echo "----------------------------------------------------------------"
