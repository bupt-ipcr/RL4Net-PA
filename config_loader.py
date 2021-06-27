#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 2021-04-19 22:01
@edit time: 2021-04-21 17:43
@file: /workspace/config_loader.py
@desc: 
"""
import copy
from datetime import datetime
import itertools
import importlib
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pprint

import yaml

__ALL__ = ['get_args']


def get_args(path: Path):
    # 从配置文件中获取配置表，并补全信息
    config_dict = complete_config(get_config(path))
    # 使用完整的配置表初始化wrapper
    wrapper = ConfigWrapper(config_dict)
    # 基于wrapper生成argparse
    args = create_argparser(wrapper)
    # 基于argparse（命令行输入）更新wrapper
    parse_args(wrapper, args)
    # 获取实际的args并返回
    args = wrapper.get_args()
    return args


def get_config(path: Path):
    config = None
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _complete_value(name, value):
    if isinstance(value, dict):
        default = value.get("default", value.get("value", None))
        if default is None:
            raise ValueError(f"{value} have no default or value.")
    else:
        default = value

    c_value = {
        "default": default,
        "value": default,
        "type": type(default),
        "name": name
    }
    if isinstance(value, dict):
        c_value.update(value)

    # 为type引入特殊判断
    if c_value.get("type", None) != "class":
        c_value.update({"type": type(default)})

    return c_value


def _import_class(class_path: str):
    paths = class_path.split('.')
    module_path = '.'.join(paths[:-1])
    class_name = paths[-1]
    module = importlib.import_module(module_path)
    clz = getattr(module, class_name)
    return clz


def complete_config(config: dict):
    """将支持简便写法的config补全"""
    completed_config = {}

    # scopes 是 config 文件的根目录，如果找不到应当直接报错
    scope_briefs, alias_groups = {}, []
    scopes = config["scopes"]
    for scope in scopes:
        c_configs = {}

        configs = scope["configs"]
        for name, val in configs.items():
            # 将complete之后的value设置到c_configs
            completed_value = _complete_value(name, val)
            c_configs[name] = completed_value

            # 如果存在alias，添加到alias_group
            if 'alias' in completed_value:
                # 初始化一个alias_group，为当前name添加对应的scope前缀
                alias_group = {scope["name"] + '@' + name}
                # 将所有alias的名字添加到当前alias_group
                # 还要将参数复制一份给alias的scope@name
                for a_name in completed_value['alias']:
                    assert '@' in a_name
                    alias_group.add(a_name)
                    a_scope, a_name = a_name.split('@')
                    completed_config.setdefault(a_scope, {})[
                        a_name] = completed_value.copy()
                alias_groups.append(alias_group)

        scope_name = scope["name"]
        scope_brief = scope["brief"] if "brief" in scope else scope["name"][0]
        scope_briefs[scope_brief] = scope_name
        completed_config.setdefault(scope_name, {}).update(c_configs)

    # 设置meta属性
    completed_config['scopes'] = scope_briefs
    completed_config['alias_groups'] = alias_groups

    # 确保main的存在
    if "main" not in completed_config:
        completed_config["main"] = {}
        completed_config['scopes'].update({'m': 'main'})

    return completed_config


class ConfigWrapper:
    def __init__(self, _d):
        """将所有scope以及原本的meta信息作为config对象的attributes"""
        d = copy.deepcopy(_d)
        self.__dict__.update(d)

    def get(self, name):
        """
        针对配置文件的get方法，按照如下顺序查找：
        1. 如果name是使用@分割的，则在特定的scope里查找
        2. 如果没有使用@分割，优先在self 的属性（即__dict__）查找
        3. 最后尝试在self 的 main scope查找
        Exception:
            如果上述查找都失败了，抛出AttributeError
        """
        if '@' in name:
            scope, name = name.split('@')
            return getattr(self, scope)[name]
        elif hasattr(self, name):
            return getattr(self, name)
        elif name in self.main:
            return self.main[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' "
                                 f"object has no attribute '{name}'")

    def set(self, name, value):
        """
        针对配置文件的set方法，按照如下规则设置：
        1. 如果原本就有这个属性，就直接设置属性（同__setattr__）
        2. 如果原本没有这个属性，则认为一定是对scope的设置，补齐scope信息
            并将对应属性的value值设为value
        2.1. 如果name在alias_group中，则同步更新整个group
        2.2. 如果不在，则只更新在特定scope下的值
        """
        if hasattr(self, name):
            setattr(self, name, value)
            return

        if '@' not in name:
            name = 'main@' + name

        for group in self.alias_groups:
            if name in group:
                for g in group:
                    self._setattr(g, value)
                return
        else:
            self._setattr(name, value)

    def _setattr(self, name, value):
        scope, name = name.split('@')
        getattr(self, scope)[name]['value'] = value

    def get_changes(self):
        """return a str contains changes join with &"""
        changes = set()
        for scope in self.scopes.values():
            configs = self.get(scope)
            for config in configs.values():
                if config.get("ignore_change", False):
                    continue
                if config['value'] != config['default']:
                    if config['type'] != list:
                        change = str(config['value'])
                    elif config['type'] == 'class':
                        change = config['value'].__class__.__name__
                    else:
                        change = '+'.join(map(str, config['value']))
                    changes.add(config['name'] + '=' + change)
        return '&'.join(changes) if changes else ''

    def get_logdir(self):
        changes = self.get_changes()
        rootdir = Path('runs')  / (changes if changes else 'default')
        seed = self.env['seed']['value']
        seeddir = f'seed={seed}'

        now = datetime.now()
        nowdir = '_'.join(now.ctime().split(' ')[1:-1])

        logdir = rootdir / seeddir / nowdir
        return logdir

    def get_args(self):
        def convert_configs(_c):
            c = {}
            for k, v in _c.items():
                if v['type'] != 'class':
                    c.update({k: v['value']})
                else:
                    v_ = _import_class(v['value'])
                    c.update({k: v_})
            return c

        args = Namespace()
        for scope in self.scopes.values():
            configs = convert_configs(self.get(scope))
            if scope == 'main':
                for k, v in configs.items():
                    setattr(args, k, v)
            else:
                setattr(args, scope, configs)

        # 为args设置logdir这个属性
        args.logdir = self.get_logdir()

        return args

    def to_dict(self):
        return self.__dict__.copy()


def create_argparser(wrapper: ConfigWrapper):
    parser = ArgumentParser(
        description="""Auto generated argument parser from config file.
        For scope changes, input like 'NAME=VALUE';
        if new value is a list, use '+' to join like 'V1+V2+V3'."""
    )

    # main scope 的参数直接添加到主参数列表
    main_group = parser.add_argument_group('main', "main group of args")
    main_scope = wrapper.get('main')
    for name, value in main_scope.items():
        # 如果value中存在brief字段，则这个arg有短前缀；否则没有
        names = ['--' + name]
        if "brief" in value:
            names.append('-'+value["brief"])

        main_group.add_argument(
            *names,  # 将names解包作为add_argument的所有的name输入
            dest=name,  # 以name为目标变量名称dest
            # 如果type给出的是class，则命令行应该接收其path作为输入，所以是str
            type=value['type'] if value['type'] != 'class' else str,
            default=value['value']  # 以配置文件中的value为准，避免修改
        )

    # 其他的scope参数以键值对的形式添加
    for brief, name in wrapper.get('scopes').items():
        parser.add_argument(
            '-'+brief, '--'+name+'_changes',  # names
            dest=name+'_changes',
            type=str,
            action='append',
            default=[], nargs='+',
            help=f'Changes in {name} scope.'
        )
    args = parser.parse_args()
    return args


def parse_args(wrapper: ConfigWrapper, args: Namespace):
    main_args, scope_args = {}, {}
    # 将所有参数归类
    for k, v in args.__dict__.items():
        if k.endswith('_changes'):
            # 注意不能用rstrip，rstrip以字符为单位删除
            name = k[: -len('_changes')]
            scope_args[name] = v
        else:
            main_args[k] = v

    # 处理main scope 的参数
    for k, v in main_args.items():
        name = 'main@' + k
        wrapper.set(name, v)

    # 处理其他scope的参数
    for scope, scope_arg in scope_args.items():
        for arg in itertools.chain(*scope_arg):
            k, v = arg.split('=')
            name = scope + '@' + k
            # 判断name的原始类型是不是支持'+'分割
            origin = wrapper.get(name)
            if origin['type'] == list:
                wrapper.set(name, v.split('+'))
            else:
                wrapper.set(name, v)


if __name__ == '__main__':
    args = get_args('config.yaml')
    DQN = args.DQN
    agent = DQN()
