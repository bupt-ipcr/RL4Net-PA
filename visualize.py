from operator import and_
from functools import reduce
from inspect import isfunction
import json
import re

from matplotlib.pyplot import plot
import utils
import config_loader
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
valid_keys = ['Number of RXs in each cluster', 'Number of TXs',
              'Number of CUs', 'BS Power (W)']
alias = {
    'bs_power': 'BS Power (W)', 'm_usrs': 'Number of CUs',
    'n_t_devices': 'Number of TXs',
    'm_r_devices': 'Number of RXs in each cluster'
}
HUE_ORDER = ['DRPA', 'FP', 'WMMSE', 'maximum', 'random']
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
    for name, func in plot_funcs.items():
        parser.add_argument(f'--{name}', action='store_true',
                            help=f'Whether to plot {name}.')
    args = parser.parse_args()
    if not any(arg[1] for arg in args._get_kwargs() if arg[0] not in {'dir', 'reload'}):
        args.all = True
    return args


def get_default_config(rename=False):
    c_config = config_loader.complete_config(config_loader.get_config("config.yaml"))
    env_config = c_config['env']
    config = {
        k: v["default"]
        for k, v in env_config.items()
    }
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
                      hue_order=HUE_ORDER,
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
                     hue_order=HUE_ORDER,
                     height=3, aspect=1.5, facet_kws=dict(legend_out=False),
                     # aspect=1.5, facet_kws=dict(legend_out=False),
                     **kwargs)
    ax.legend.set_title('')
    ax.legend._loc = 7
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
                     hue_order=HUE_ORDER,
                     showfliers=False, **kwargs)
    ax.legend().set_title('')
    plt.ylabel(f'Average {aim} (bps/Hz)')
    ax.grid(axis="y")
    return fig, ax


def check_and_savefig(path: Path(), *args, **kwargs):
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    plt.savefig(path, *args, **kwargs)


def get_all_data(args):
    runsdir = here / args.dir
    # try to load data from pickle
    save_file = here / args.file
    with save_file.open('rb') as f:
        all_data = pickle.load(f)
        print('Load data from pickle.')
        # filter
        all_data = all_data[all_data["bs_power"]>=6]
        all_data = all_data[all_data["bs_power"]<=11]
        all_data = all_data[all_data["m_usrs"]<=6]
        all_data = all_data[all_data["algorithm"]!="D3QN"]
        # rename
        # all_data.rename(columns={
        #     "DQN": "DRPA"
        # },inplace=True)
        all_data.loc[all_data["algorithm"]=="DQN", "algorithm"]="DRPA"
        all_data.rename(columns=alias,
                inplace=True)
        all_data.rename(columns={
            'Number of CUE': 'Number of CUs', 
            'Number of DTs': 'Number of TXs',
            'Number of DRs in each cluster': 'Number of RXs in each cluster',
        },inplace=True)
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
    ) if k in valid_keys and k not in {'Number of CUs', 'BS Power', 'Sum BS Power'}))
    key = 'Sum BS Power'
    for aim in tqdm(['Rate', 'sum-rate'], desc='Ploting SBP'):
        fig = plt.figure(figsize=(15, 10))
        sns.boxplot(x=key, y=aim, hue="algorithm", hue_order=HUE_ORDER,
                    data=all_data[cur_index], palette="Set1", showfliers=False)
        check_and_savefig(figs / f'box/{aim}-{key}.png')
        plt.close(fig)

        fig = plt.figure(figsize=(15, 10))
        sns.lineplot(data=all_data[cur_index], x=key, y=aim, hue="algorithm",
                     hue_order=HUE_ORDER,
                     style="algorithm", markers=True, dashes=False, ci=None)
        plt.xticks(sorted(list(set(all_data[cur_index][key]))))
        check_and_savefig(figs / f'avg/{aim}-{key}.png')
        plt.close(fig)


@register
def plot_globe(all_data):
    aim, palette = "sum-rate", 'Set1'
    # missions
    missions = [('CDF', displot), ('BS Power (W)', lineplot),
                ('Number of CUs', lineplot), ('Number of RXs in each cluster', boxplot),
                ('Number of TXs', boxplot)]
    for mission in tqdm(missions, desc="Ploting GlobeCom"):
        key, func = mission

        fig, ax = func(data=all_data, key=key, aim=aim,
                       palette=sns.color_palette(palette, len(HUE_ORDER)))
        if func == lineplot:
            ax.set_ylim((15, 90))
        elif func == boxplot:
            ax.set_ylim((0, 140))
        check_and_savefig(figs / f'globe/{aim}-{key}-{palette}.png',
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
