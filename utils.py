import json
from datetime import datetime
from pathlib import Path
from functools import wraps
import numpy as np
import yaml
from argparse import ArgumentParser, Namespace
import config_loader
config_path = 'config.yaml'
default_config_path = 'default_config.yaml'


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = datetime.now()
        ret = func(*args, **kwargs)
        end = datetime.now()
        print(f'<{func.__name__}> cost time:', end - start)
        return ret
    return wrapper


def create_seeds():
    init_seed = (19980615 * 19970711) % 1059999
    seed_count = 100
    np.random.seed(init_seed)
    seeds = list(np.random.randint(9999, 1059999, seed_count))
    save_path = Path('seed.json')
    with save_path.open('w') as f:
        json.dump(str(seeds), f)
    return seeds


def check_exist(logdir):
    # check if logdir has result.log
    parent = logdir.parent
    if parent.exists():
        for train_dir in parent.iterdir():
            for train_file in train_dir.iterdir():
                if train_file.name == "results.log":
                    return True
        # clear
        for train_dir in parent.iterdir():
            for train_file in train_dir.iterdir():
                train_file.unlink()
            try:
                train_dir.rmdir()
            except:
                print(f'{train_dir}.rmdir() failed')
            print(f'{train_dir}.rmdir()')
    return False


def get_args_from_config(config_path: Path)->Namespace():
    args = config_loader.get_args(config_path)
    return args