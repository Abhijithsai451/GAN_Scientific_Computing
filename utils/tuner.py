import os
import yaml
import subprocess


def run_experiment(lr_g, lr_d, batch_size, experiment_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    config_dir = os.path.join(root, 'config')
    os.makedirs(config_dir, exist_ok=True)
    temp_config_path = os.path.join(config_dir, f"temp_config_{experiment_name}.yaml")
    # Load base config
    with open('config/improved_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Updating the hyperparameters
    config['project_name'] = experiment_name
    config['trainer']['lr_g'] = lr_g
    config['trainer']['lr_d'] = lr_d
    config['trainer']['batch_size'] = batch_size
    config['logger']['log_dir'] = f"results/tuning/{experiment_name}/logs"


    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"--- Running Experiment: {experiment_name} ---")
    subprocess.run(["python", "main.py", "--config", temp_config_path])
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)


if __name__ == "__main__":
    lrs = [0.0001, 0.0002, 0.0005]
    for lr in lrs:
        run_experiment(lr, lr, 64, f"search_lr_{lr}")