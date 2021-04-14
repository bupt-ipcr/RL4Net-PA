#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-07 20:17
@edit time: 2021-04-14 16:28
@file: /RL4Net-PA/policy_dqn.py
"""
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from rl4net.agents.DQN_base import DQNBase
import os

CUDA = torch.cuda.is_available()


class PADQNNet(nn.Module):
    """一个只有一层隐藏层的DQN神经网络"""

    def __init__(self, n_states, n_actions, card_no=0):
        """
        定义隐藏层和输出层参数
        @param n_obs: number of observations
        @param n_actions: number of actions
        @param n_neurons: number of neurons for the hidden layer
        """
        super(PADQNNet, self).__init__()
        self.card_no = card_no
        self.seq = nn.Sequential(
            nn.Linear(n_states, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        定义网络结构: 第一层网络->ReLU激活->输出层->softmax->输出
        """
        if CUDA:
            x = x.cuda(self.card_no)
        action_values = self.seq(x)
        return action_values


class DQN(DQNBase):
    """
    基于DQNBase创建的DQN类，通过附带的简单神经网络创建了 eval dqn network 和 target dqn network
    """

    def __init__(self, n_states, n_actions, **kwargs):
        super().__init__(n_states, n_actions, **kwargs)

    def _build_net(self):
        self.eval_net = PADQNNet(self.n_states, self.n_actions, self.card_no)
        self.target_net = PADQNNet(self.n_states, self.n_actions, self.card_no)

    def add_steps(self, cur_state, action, reward, done, next_state):
        size = action.shape[0]
        for i in range(size):
            self.add_step(cur_state[i], action[i],
                          reward, done, next_state[i])

    def get_action(self, state):
        # 将行向量转为列向量（1 x n_states -> n_states x 1 x 1)
        if np.random.rand() < self.epsilon:
            # 概率随机
            action_size = state.shape[0]
            return np.random.randint(0, self.n_actions, (1, action_size))
        else:
            # greedy
            state = torch.FloatTensor(state)
            state = torch.unsqueeze(torch.FloatTensor(state), 0)
            action_values = self.eval_net.forward(state).cpu()
            return action_values.data.numpy().argmax(axis=2)

    def save(self, episode=None, save_path='./cur_model.pth', append_dict={}):
        """Save the network parameters of the current model.

        Args:
          save_path: The save path of the model.
          append_dict: What needs to be saved in addition to the network model.
        """
        states = {
            'eval_net': self.eval_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'episode': self.episode if episode is None else episode,
            'eval_step': self.eval_step
        }
        states.update(append_dict)
        torch.save(states, save_path)


    def _load(self, save_path):
        """Load the model parameter.

        Args:
          save_path: The save path of the model.

        Returns:
          The loaded model dictionary.
        """
        if CUDA:
            states = torch.load(save_path, map_location=torch.device('cuda'))
        else:
            states = torch.load(save_path, map_location=torch.device('cpu'))

        # load network parameters from the model
        self.eval_net.load_state_dict(states['eval_net'])
        self.target_net.load_state_dict(states['target_net'])

        # load 'episode' and 'step' from the model
        self.episode, self.eval_step = states['episode'], states['eval_step']
        # return states
        return states

    def load(self, save_path='./cur_model.pth'):
        """The default implementation of the loaded model.

        Args:
          save_path: The save path of the model，default is'./cur_model.pth'.

        Returns:
          Recorded episode value.
        """
        print('\033[1;31;40m{}\033[0m'.format('加载模型参数...'))
        if not os.path.exists(save_path):
            print('\033[1;31;40m{}\033[0m'.format('没找到保存文件'))
            return -1
        else:
            states = self._load(save_path)
            return states['episode']
