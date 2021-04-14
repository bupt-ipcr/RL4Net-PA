#!/usr/bin/env python
# coding=utf-8
"""
@author: Jiawei Wu
@create time: 2019-12-06 23:01
@edit time: 2021-04-14 16:09
@file: /RL4Net-PA/pa_ddpg.py
"""

import torch
import numpy as np
import time
import gym
from rl4net.agents import DDPGBase
from rl4net.models import SimpleActorNet, SimpleCriticNet
from rl4net.envs.power_allocation import PAEnv, Node
from rl4net.utils.OUProcess import OUProcess
CUDA = torch.cuda.is_available()


class DDPG(DDPGBase):
    """DDPG类创建示例"""
    def _param_override(self):
        self.summary = True
        self.buff_size=10000
        self.buff_thres=2000
        self.batch_size=128
        
    def _build_net(self):
        n_states, n_actions = self.n_states, self.n_actions
        self.actor_eval = SimpleActorNet(n_states, n_actions, a_bound=self.bound)
        self.actor_target = SimpleActorNet(n_states, n_actions, a_bound=self.bound)
        self.critic_eval = SimpleCriticNet(n_states, n_actions)
        self.critic_target = SimpleCriticNet(n_states, n_actions)

    def _build_noise(self):
        self.noise = OUProcess(self.n_actions, sigma=0.1)
        # 当不需要nosie函数的特殊情况，可以略过
        pass 

    def get_action(self, s):
        """展平state"""
        s = s.reshape((1, self.n_states))
        return self._get_action(s)

    def get_action_noise(self, state, rate=1):
        action = self.get_action(state)
        action_noise = np.clip(np.random.normal(0, 3), -2, 2) * rate
        action += action_noise
        return action[0]
    
    def add_step(self, s, a, r, d, s_):
        self._add_step(s, a, r, d, s_)


def rl_loop():
    MAX_EPISODES = 1000

    env = PAEnv(n_levels=10, n_t_devices=25, m_r_devices=4, m_usrs=0, R_bs=5, R_dev=1.)
    s_dim = env.n_states * env.n_actions
    a_dim = env.n_actions
    a_bound = env.m_actions

    ddpg = DDPG(n_states=s_dim, n_actions=a_dim, bound=a_bound)
    t1 = time.time()
    rate = 1
    for i in range(MAX_EPISODES):
        s = env.reset()
        ep_reward = 0
        while True:
            # Add exploration noise
            a = ddpg.get_action_noise(s, rate)[0]
            a = a.astype(np.int8).clip(0, a_bound-1)
            s_, r, done, info = env.step(a)
            r = sum(r)
            ddpg.add_step(s, a, r / 10, done, s_)

            if ddpg.learn():
                rate *= .9995    # decay the action randomness

            s = s_
            ep_reward += r/env.n_actions
            if done:
                print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % rate, )
                ddpg.summary_writer.add_scalar('ep_reward', ep_reward, i)
                break

    print('Running time: ', time.time() - t1)


if __name__ == '__main__':
    rl_loop()
