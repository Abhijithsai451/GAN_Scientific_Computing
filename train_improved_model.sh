#!/bin/bash

# Configuration
REPO_URL="https://github.com/Abhijithsai451/GAN_Scientific_Computing.git"
CONFIG_FILE="improved_config.yaml"

echo "----------------------------------------------------------------"
echo "Starting Execution: Improved Model"
echo "----------------------------------------------------------------"

echo "Pulling latest code from GitHub..."
if [ -d ".git" ]; then
    git pull origin master
else
    echo "Git repository not found. Cloning..."
    git clone $REPO_URL .
fi

# 2. Install requirements
echo "Installing dependencies from requirements.txt..."
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
TUNER_SCRIPT="utils/tuner.py"

echo "================================================================"
echo "Hyperparameter Tuning (Grid Search)"
echo "================================================================"
python3 $TUNER_SCRIPT
# 3. Create necessary directories
echo "Ensuring directory structure exists..."
mkdir -p data results/improved/logs results/improved/runs results/improved/checkpoints

# 4. Run the model
echo "================================================================"
echo "STAGE 2: Final Training with Optimized Parameters"
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
