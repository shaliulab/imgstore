import os.path
import yaml
from imgstore.constants import CONFIG_FILE

def load_config():

    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as filehandle:
            config = yaml.load(filehandle, yaml.SafeLoader)
    else:
        config = {"codecs": {}}

    return config


def save_config(config):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w") as filehandle:
        yaml.dump(config, filehandle, yaml.SafeDumper)
