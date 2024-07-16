import toml
from easydict import EasyDict
import os
def get_config():
    with open(os.path.join('config.toml'), 'r', encoding='utf-8') as f:
        config = toml.load(f)
        return EasyDict(config)
