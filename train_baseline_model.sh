#!/bin/bash

# Configuration
REPO_URL="https://github.com/Abhijithsai451/GAN_Scientific_Computing.git"
CONFIG_FILE="baseline_config.yaml"

echo "----------------------------------------------------------------"
echo "Starting Execution: Baseline Model Pipeline"
echo "----------------------------------------------------------------"

# 1. Pull latest code from GitHub
echo "Checking for code updates from GitHub..."
if [ -d ".git" ]; then
    git pull origin master
else
    echo "Git repository not found. Cloning..."
    git clone $REPO_URL .
fi

# 2. Install requirements
echo "Ensuring dependencies are installed..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 3. Create necessary directories for Baseline
echo "Ensuring directory structure exists..."
mkdir -p data results/baseline/logs results/baseline/runs results/baseline/checkpoints

# 4. Stage 1: Main Training
echo "================================================================"
echo "Baseline Model Training"
echo "================================================================"
# Running the model using the baseline configuration
python3 main.py --config $CONFIG_FILE

# 5. Stage 2: Evaluation & Testing
echo "================================================================"
echo "Evaluation & Testing"
echo "================================================================"
# Running the test script on the baseline checkpoints
python3 test.py --config $CONFIG_FILE

echo "----------------------------------------------------------------"
echo "Execution Finished Successfully."
echo "Check 'results/logs' for curves and 'results/test_results' for baseline grids."
echo "----------------------------------------------------------------"