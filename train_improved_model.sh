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
    git pull origin Protected
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
TUNER_SCRIPT="wandb_utils/wandb_tuner.py"
# 4. Run Execution
if [ "$TUNE_MODEL" = true ]; then
    echo "================================================================"
    echo "Automated Hyperparameter Tuning (W&B Sweep)"
    echo "================================================================"
    # We call the Tuner script instead of main.py
    python3 $TUNER_SCRIPT
else
  echo ">>> Skipping Hyperparameter Tuning. Using existing $CONFIG_FILE."
fi

# 4. Run the model
echo "================================================================"
echo "Training with Optimized Parameters"
echo "================================================================"
python3 main.py --config $CONFIG_FILE

echo "================================================================"
echo "Evaluation & Testing"
echo "================================================================"
# Run the test script on the final checkpoints
python3 test.py --config $CONFIG_FILE

echo "----------------------------------------------------------------"
echo "Execution Finished."
echo "Check 'results/improved/logs' for curves and 'results/test_results' for grids."
echo "----------------------------------------------------------------"
