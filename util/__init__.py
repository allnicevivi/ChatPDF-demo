
import yaml
import os

config_path = os.path.join(os.getcwd(), "util\config.yaml")

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
