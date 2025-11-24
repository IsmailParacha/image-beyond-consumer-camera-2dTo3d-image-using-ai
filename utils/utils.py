# utils/utils.py (Needed to load config)
import yaml
def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)