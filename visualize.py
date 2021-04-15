from operator import and_
from functools import reduce
from inspect import isfunction
import json
import re

from matplotlib.pyplot import plot
import utils
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm, trange
import pickle

here = Path()
figs = here / 'figs'
valid_keys = ['Number of DRs in each cluster', 'Number of DTs',
              'Number of CUE', 'BS Power (W)']
alias = {
    'bs_power': 'BS Power (W)', 'm_usrs': 'Number of CUE',
    'n_t_devices': 'Number of DTs',
    'm_r_devices': 'Number of DRs in each cluster'
}
plot_funcs = {}


def register(func):
    plot_funcs[func.__name__[5:]] = func
    return func


def get_args():
    parser = ArgumentParser(description='Params for ploting.')
    parser.add_argument('-d', '--dir', type=str, default='runs',
                        help='Directory to visualize.')
    parser.add_argument('-f', '--file', type=str, default='all_data.pickle',
                        help='File of all_data.')
    parser.add_argument('-r', '--reload', action='store_true',
                        help='Force to reload data.')
    for name, func in plot_funcs.items():
        parser.add_argument(f'--{name}', action='store_true',
                            help=f'Whether to plot {name}.')
    args = parser.parse_args()
    if not any(arg[1] for arg in args._get_kwargs() if arg[0] not in {'dir', 'reload'}):
        args.all = True
    return args


def get_default_config(rename=False):
    dft_config = utils.get_config('default_config.yaml')
    config = dft_config['env']
    agent_config = dft_config['agent']
    config.update(agent_config)
    for k, v in config.items():
        if isinstance(v, list):
            config[k] = '+'.join(v)
        try:
            config[k] = int(v)
        except:
            pass
    # add
    if rename:
        for k, v in alias.items():
            config[v] = config[k]
    return config


dft_config = get_default_config(rename=True)


def lineplot(data, key, aim, **kwargs):
    sns.set_style('whitegrid')
    # fig = plt.figure(figsize=(10, 7.5))
    fig = plt.figure()
    cur_index = reduce(and_, (all_data[k] == v for k, v in dft_config.items(
    ) if k in valid_keys and k != key))
    plt.xticks(sorted(list(set(data[key]))))
    ax = sns.lineplot(data=data[cur_index], x=key, y=aim, hue="algorithm",
                      hue_order=['FP', 'DRPA', 'WMMSE', 'maximum', 'random'],
                      style="algorithm", markers=True, dashes=False, ci=None,
                      markersize=8, **kwargs)
    ax.legend().set_title('')
    plt.ylabel(f'Average {aim} (bps/Hz)')
    return fig, ax


def displot(data, key, aim, **kwargs):
    sns.set_style('white')
    # fig = plt.figure(figsize=(10, 7.5))
    fig = plt.figure()
    ax = sns.displot(data=data, x=aim, kind="ecdf", hue="algorithm",
                     hue_order=['FP', 'DRPA', 'WMMSE', 'maximum', 'random'],
                     height=3, aspect=1.5, facet_kws=dict(legend_out=False),
                    # aspect=1.5, facet_kws=dict(legend_out=False),
                     **kwargs)
    ax.legend.set_title('')
    ax.legend._loc=7
    plt.xlabel(f'Average {aim} (bps/Hz)')
    plt.grid(axis="y")
    return fig, ax


def boxplot(data, key, aim, **kwargs):
    sns.set_style('white')
    # fig = plt.figure(figsize=(10, 7.5))
    fig = plt.figure()
    cur_index = reduce(and_, (all_data[k] == v for k, v in dft_config.items(
    ) if k in valid_keys and k != key))
    plt.xticks(sorted(list(set(data[key]))))
    ax = sns.boxplot(data=data[cur_index], x=key, y=aim, hue="algorithm",
                     hue_order=['FP', 'DRPA', 'WMMSE', 'maximum', 'random'],
                     showfliers=False, **kwargs)
    ax.legend().set_title('')
    plt.ylabel(f'Average {aim} (bps/Hz)')
    ax.grid(axis="y")
    return fig, ax


def check_and_savefig(path: Path(), *args, **kwargs):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path, *args, **kwargs)


def get_datas(directory: Path()):
    datas = []
    for path in directory.iterdir():
        if path.is_dir():
            sub_datas = get_datas(path)
        elif path.name == 'results.log':
            sub_datas = []
            with path.open() as f:
                for line in f.readlines():
                    if re.match(r'^[a-z]+: [\d.]+$', line):
                        algorithm, rate = line.split(': ')
                        rate = float(rate)
                        algorithm_mapping = {
                            'dqn': 'DRPA',
                            'fp': 'FP',
                            'wmmse': 'WMMSE',
                            'random': 'random',
                            'maximum': 'maximum'
                        }
                        sub_datas.append({
                            'algorithm': algorithm_mapping[algorithm],
                            'Rate': rate
                        })
        else:
            sub_datas = []
        datas.extend(sub_datas)
    return datas


def get_all_data(args):
    runsdir = here / args.dir
    # try to load data from pickle
    save_file = here / args.file
    if not args.reload and save_file.exists():
        with save_file.open('rb') as f:
            all_data = pickle.load(f)
            print('Load data from pickle.')
            return all_data
    config = get_default_config()
    all_data = []
    for logdir in tqdm(list(runsdir.iterdir()), desc="Gathering all data"):
        conf = config.copy()
        n_recvs = conf['n_t_devices'] * conf['m_r_devices'] +\
            conf['n_bs'] * conf['m_usrs']
        if logdir.name != 'default':
            changes = logdir.name.split('&')
            for change in changes:
                key, value = change.split('=')
                if key != 'card_no':
                    try:
                        conf[key] = int(value)
                    except:
                        try:
                            conf[key] = float(value)
                        except:
                            conf[key] = value
        datas = get_datas(logdir)
        for data in datas:
            data.update(conf)
            data['sum-rate'] = data['Rate'] * n_recvs
        if not 2 < conf['bs_power'] < 40:
            continue
        all_data.extend(datas)
    # save
    all_data = pd.DataFrame(all_data)
    # replace column names
    all_data.rename(columns=alias,
                    inplace=True)
    all_data.to_pickle(str(save_file))
    return all_data


@register
def plot_avg(all_data):
    for key in tqdm(valid_keys, desc="Ploting AVG"):
        for aim in ['Rate', 'sum-rate']:
            fig = lineplot(data=all_data, key=key, aim=aim)
            check_and_savefig(figs / f'avg/{aim}-{key}.png')
            plt.close(fig)


@register
def plot_box(all_data):
    for key in tqdm(valid_keys, desc="Ploting Box"):
        for aim in ['Rate', 'sum-rate']:
            fig = boxplot(data=all_data, key=key, aim=aim)
            check_and_savefig(figs / f'avg/{aim}-{key}.png')
            plt.close(fig)
            check_and_savefig(figs / f'box/{aim}-{key}.png')
            plt.close(fig)


@register
def plot_cdf(all_data):
    for aim in tqdm(['Rate', 'sum-rate'], desc="Ploting CDF"):
        fig = displot(data=all_data, key='', aim=aim)
        check_and_savefig(figs / f'cdf/{aim}.png')
        plt.close(fig)


@register
def plot_sbp(all_data):
    """Plot sum bs power"""
    all_data['Sum BS Power'] = all_data['BS Power'] * all_data['m_usrs']
    cur_index = reduce(and_, (all_data[k] == v for k, v in dft_config.items(
    ) if k in valid_keys and k not in {'Number of CUE', 'BS Power', 'Sum BS Power'}))
    key = 'Sum BS Power'
    for aim in tqdm(['Rate', 'sum-rate'], desc='Ploting SBP'):
        fig = plt.figure(figsize=(15, 10))
        sns.boxplot(x=key, y=aim, hue="algorithm", hue_order=['FP', 'DRPA', 'WMMSE', 'maximum', 'random'],
                    data=all_data[cur_index], palette="Set1", showfliers=False)
        check_and_savefig(figs / f'box/{aim}-{key}.png')
        plt.close(fig)

        fig = plt.figure(figsize=(15, 10))
        sns.lineplot(data=all_data[cur_index], x=key, y=aim, hue="algorithm",
                     hue_order=['FP', 'DRPA', 'WMMSE', 'maximum', 'random'],
                     style="algorithm", markers=True, dashes=False, ci=None)
        plt.xticks(sorted(list(set(all_data[cur_index][key]))))
        check_and_savefig(figs / f'avg/{aim}-{key}.png')
        plt.close(fig)


@register
def plot_env(*args):
    seed = 312691
    env = utils.get_env(seed=seed, n_t_devices=3,
                        m_r_devices=4, m_usrs=4, R_dev=0.25)
    import matplotlib.patches as mpatches

    def cir_edge(center, radius, color):
        patch = mpatches.Circle(center, radius, fc='white', ec=color, ls='--')
        return patch

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # draw d2d pairs
    for t_idx, pair in env.devices.items():
        t, rs = pair['t_device'], pair['r_devices']
        # draw edge
        ax.add_patch(cir_edge((t.x, t.y), env.R_dev, 'green'))
        # draw t device
        ax.scatter([t.x], [t.y], marker='s', s=100, c='green', zorder=10)
        # draw r devices
        for r_idx, r in rs.items():
            ax.scatter([r.x], [r.y], marker='o', s=60, c='green', zorder=10)

    # draw cell and bs
    cell_xs = env.R_bs * \
        np.array([0, np.sqrt(3)/2, np.sqrt(3)/2,
                  0, -np.sqrt(3)/2, -np.sqrt(3)/2, 0])
    cell_ys = env.R_bs * np.array([1, .5, -.5, -1, -.5, .5, 1])
    ax.plot(cell_xs, cell_ys, color='black')

    ax.scatter([0.0], [0.0], marker='^', s=100, c='blue', zorder=30)
    # draw usrs
    for idx, usr in env.users.items():
        ax.scatter([usr.x], [usr.y], marker='x', s=100, c='orange', zorder=20)
        ax.plot([0, usr.x], [0, usr.y], ls='--', c='blue', zorder=20)

    check_and_savefig(figs / f'env/{seed}.png')
    plt.close(fig)


@register
def plot_icc(all_data):
    aim, palette = "sum-rate", 'Set1'
    # missions
    missions = [('CDF', displot), ('BS Power (W)', lineplot),
                ('Number of CUE', lineplot), ('Number of DRs in each cluster', boxplot),
                ('Number of DTs', boxplot)]
    for mission in tqdm(missions, desc="Ploting ICC"):
        key, func = mission

        fig, ax = func(data=all_data, key=key, aim=aim,
                       palette=sns.color_palette(palette, 5))
        if func == lineplot:
            ax.set_ylim((10, 80))
        elif func == boxplot:
            ax.set_ylim((0, 140))
        check_and_savefig(figs / f'icc/{aim}-{key}-{palette}.png',
                          dpi=300)
        plt.close(fig)


@register
def plot_all(all_data):
    for name, func in plot_funcs.items():
        if name != 'all':
            func(all_data)


if __name__ == "__main__":
    args = get_args()
    all_data = get_all_data(args)
    for attr in dir(args):
        if not attr.startswith('_') and args.__getattribute__(attr):
            func = plot_funcs.get(attr, None)
            if func:
                func(all_data)
