import json
from datetime import datetime
from pathlib import Path
from functools import wraps
import numpy as np
import yaml
from rl4net.envs.power_allocation import PAEnv_v0, PAEnv_v1, PAEnv_v2, PAEnv_v3
from argparse import ArgumentParser
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


def get_args():
    parser = ArgumentParser()
    parser.add_argument('-e', '--env_changes', type=str,
                        action='append',
                        default=[], nargs='+',
                        help='Changes of env compare to default config.')
    parser.add_argument('-a', '--agent_changes', type=str,
                        action='append',
                        default=[], nargs='+',
                        help='Changes of agent compare to default config.')
    parser.add_argument('-c', '--card_no', type=int,
                        help='GPU card no.', default=0)
    parser.add_argument('-o', '--offset', type=int,
                        help='Seed offset.', default=0)
    parser.add_argument('-s', '--seeds', type=int,
                        help='Seed count.', default=100)
    parser.add_argument('-i', '--ignore', action='store_true',
                        help='Ignore processed seed.', default=False)
    args = parser.parse_args()
    env = {}
    for changes in args.env_changes:
        for change in changes:
            key, value = change.split('=')
            try:
                value = int(value)
            except:
                try:
                    value = json.loads(value)
                except:
                    pass
            env[key] = value
    args.env = env
    agent = {}
    # card_no
    agent['card_no'] = args.card_no
    for changes in args.agent_changes:
        for change in changes:
            key, value = change.split('=')
            try:
                value = int(value)
            except:
                try:
                    value = json.loads(value)
                except:
                    pass
            agent[key] = value
    args.agent = agent

    return args


def get_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        return config


def get_env(**kwargs):
    """默认使用config.yaml，但是kwargs优先级更高"""
    config = get_config(config_path)
    env_config = config['env']
    env_config.update(kwargs)
    env = PAEnv_v1(**env_config)
    # print(f'env.seed is {env.seed}')
    return env


def get_logdir(c1=None, c2=None) -> Path():
    config = c1 if c1 else get_config(config_path)
    default_config = c2 if c2 else get_config(default_config_path)
    env_diffs = {
        k: v for k, v in config['env'].items()
        if config['env'].get(k) != default_config['env'].get(k, None)
        and k != 'seed'
    }
    agent_diffs = {
        k: v for k, v in config['agent'].items()
        if config['agent'].get(k) != default_config['agent'].get(k, None)
    }
    env_diffs.update(agent_diffs)
    diffs = env_diffs

    rootdir = ''
    for k, v in diffs.items():
        if isinstance(v, list):
            v = '+'.join(str(vi) for vi in v)
        rootdir += f'{k}={v}&'
    rootdir = rootdir[:-1] if rootdir else 'default'

    rootdir = Path('runs') / rootdir
    if not rootdir.exists():
        rootdir.mkdir(parents=True)

    seed = config['env']['seed']
    seeddir = f'seed={seed}'

    now = datetime.now()
    nowdir = '_'.join(now.ctime().split(' ')[1:-1])

    logdir = rootdir / seeddir / nowdir
    return logdir
