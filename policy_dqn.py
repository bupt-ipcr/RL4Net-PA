#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-07 20:17
@edit time: 2021-04-15 15:41
@file: /RL4Net-PA/policy_dqn.py
"""
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from rl4net.agents.DQN_base import DQNBase, DDQNBase, DuelingDQNBase, D3QNBase
import os


class PADQNNet(nn.Module):
    """一个只有一层隐藏层的DQN神经网络"""

    def __init__(self, n_states, n_actions):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param n_neurons: number of neurons for the hidden layer
        """
        super(PADQNNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),

        )

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->softmax->输出
        """
        action_values = self.seq(x)
        return action_values


class PADuelingDQNNet(nn.Module):
    """Double DQN神经网络"""

    def __init__(self, n_states, n_actions):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param n_neurons: number of neurons for the hidden layer
        """
        super(PADuelingDQNNet, self).__init__()
        self.net_state = nn.Sequential(
            nn.Linear(n_states, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
        )
        self.net_val = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),   # Q value
            nn.ReLU()
        )
        self.net_adv = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.ReLU()
        )

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->softmax->输出
        """
        x = self.net_state(x)
        q_val = self.net_val(x)
        q_adv = self.net_adv(x)
        # dueling Q value
        return q_val + q_adv - q_adv.mean(dim=1, keepdim=True)


class DQN(DQNBase):
    """基于DQNBase创建的DQN类。
    1. 对于集中训练分布执行的特点，提供add_steps的接口，将不同device的经验添加到经验回放池。
    2. 动作获取使用按照原始动作概率随机进行epsilon-greedy，增加探索效率
    """

    def _build_net(self):
        """Build a basic network."""
        self.eval_net = PADQNNet(self.n_states, self.n_actions)
        self.target_net = PADQNNet(self.n_states, self.n_actions)

    def add_steps(self, cur_state, action, reward, done, next_state):
        size = action.shape[0]
        for i in range(size):
            self.add_step(cur_state[i], action[i],
                          reward, done, next_state[i])

    def get_action(self, state):
        action_values = self._get_action(state)
        # 将行向量转为列向量（1 x n_states -> n_states x 1 x 1)
        if np.random.rand() < self.epsilon:
            # 概率随机
            action = torch.nn.Softmax(dim=2)(action_values).squeeze()
            probs = action.detach().cpu().numpy()  # choose action according to Q value
            # corresponding to pa_dqn
            return np.array([[np.random.choice(self.n_actions, p=prob) for prob in probs]])
        else:
            # greedy
            return action_values.numpy().argmax(axis=2)
    
    def load(self, save_path='./cur_model.pth'):
        """不需要提示"""
        states = self._load(save_path)
        return states['episode']


class DDQN(DDQNBase):
    """DDQN的训练逻辑和DQN不同，所以要从DDQNBase继承"""

    def _build_net(self):
        self.eval_net = PADQNNet(self.n_states, self.n_actions)
        self.target_net = PADQNNet(self.n_states, self.n_actions)

    def add_steps(self, cur_state, action, reward, done, next_state):
        size = action.shape[0]
        for i in range(size):
            self.add_step(cur_state[i], action[i],
                          reward, done, next_state[i])

    def get_action(self, state):
        action_values = self._get_action(state)
        # 将行向量转为列向量（1 x n_states -> n_states x 1 x 1)
        if np.random.rand() < self.epsilon:
            # 概率随机
            action = torch.nn.Softmax(dim=2)(action_values).squeeze()
            probs = action.detach().cpu().numpy()  # choose action according to Q value
            # corresponding to pa_dqn
            return np.array([[np.random.choice(self.n_actions, p=prob) for prob in probs]])
        else:
            # greedy
            return action_values.numpy().argmax(axis=2)
    
    def load(self, save_path='./cur_model.pth'):
        """不需要提示"""
        states = self._load(save_path)
        return states['episode']

DoubleDQN = DDQN


class DuelingDQN(DQN):
    """DuelingDQN和DQN只是在Net上不同，所以可以直接继承DQN"""
    def _build_net(self):
        self.eval_net = PADuelingDQNNet(self.n_states, self.n_actions)
        self.target_net = PADuelingDQNNet(self.n_states, self.n_actions)


class D3QN(DDQN):
    """DuelingDQN和DDQN只是在Net上不同，所以可以直接继承DDQN"""
    def _build_net(self):
        self.eval_net = PADuelingDQNNet(self.n_states, self.n_actions)
        self.target_net = PADuelingDQNNet(self.n_states, self.n_actions)