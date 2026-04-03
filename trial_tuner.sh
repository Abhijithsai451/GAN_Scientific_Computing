#!/bin/bash

# Configuration
REPO_URL="https://github.com/Abhijithsai451/GAN_Scientific_Computing.git"
CONFIG_FILE="improved_config.yaml"

TUNE_MODEL=false
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tune) TUNE_MODEL=true ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "----------------------------------------------------------------"
echo "Starting Execution: Improved Model"
echo "Enable HyperParameter Tuning: $TUNE_MODEL"
echo "----------------------------------------------------------------"

<< :
echo "Pulling latest code from GitHub..."
if [ -d ".git" ]; then
    git pull origin master
else
    echo "Git repository not found. Cloning..."
    git clone $REPO_URL .
fi
:

# 2. Install requirements
echo "Installing dependencies..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 3. Create necessary directories
mkdir -p data results/improved/logs results/improved/checkpoints

# 4. Run Execution
if [ "$TUNE_MODEL" = true ]; then
    echo "================================================================"
    echo "MLOps Pipeline: Automated Hyperparameter Tuning (W&B Sweep)"
    echo "================================================================"
    # We call the Tuner script instead of main.py
    python3 utils/wandb_tuner.py --tune
else
    echo "================================================================"
    echo "Standard Pipeline: Training with Preset Parameters"
    echo "================================================================"
    python3 main.py --config $CONFIG_FILE
fi

echo "----------------------------------------------------------------"
echo "Execution Finished."
echo "----------------------------------------------------------------"