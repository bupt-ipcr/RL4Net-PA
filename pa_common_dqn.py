#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 2021-04-14 16:10
@edit time: 2021-04-14 21:26
@file: /RL4Net-PA/pa_common_dqn.py
@desc: 
"""

import utils
import numpy as np
from policy_dqn import DQN
# from policy_el_dqn import DQNAgentAdapter as DQN
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks
from argparse import ArgumentParser
import json

args = utils.get_args()
seeds = utils.create_seeds()

MAX_EPISODES = args.seeds
DECAY_THRES = int(MAX_EPISODES*0.75)




@utils.timeit
def dqn_loop(env, agent, logdir):
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    print(f"Start DQN loop.")
    train_his, best_score = [], 0
    for ep in range(MAX_EPISODES):
        idx = ep%len(seeds)
        cur_state = env.reset(seed=seeds[idx])
        cur_state = cur_state.reshape((-1, env.n_states))
        agent.epsilon = max((DECAY_THRES - ep) / DECAY_THRES, 0.001)
        ep_his = []
        while True:
            action = agent.get_action(cur_state)[0]
            next_state, reward, done, info = env.step(
                action.astype(np.int32))
            next_state = next_state.reshape((-1, env.n_states))
            agent.add_steps(cur_state, action, reward, done, next_state)
            ep_his.append(reward)
            loss = agent.learn()
            if loss:
                summary_writer.add_scalar('loss', loss, agent.eval_step)
            cur_state = next_state
            if done:
                break
        train_his.append(np.mean(ep_his))
        # output every 10 ep
        score = np.mean(train_his[-len(seeds):])
        if ep % 10 == 0:
            if ep > len(seeds):
                benchmark = test_with_seed(env, agent, 980615)
                print(f'EP: {ep} | score: {score} | benchmark: {benchmark}', flush=True)
            else:
                print(f'EP: {ep} | score: not engouth data', flush=True)
        # try to save
        if ep > len(seeds) and score > best_score:
            best_score = score
            agent.save(episode=ep, save_path=f"{logdir / 'common_model.pth'}")

    # after train, save train info
    with open(f"{logdir / 'train_his.json'}", 'x') as f:
        json.dump(train_his, f)
    # test model ability
    test_score = np.mean([test_with_seed(env, agent, seed) for seed in seeds])
    print('test_score is :', test_score)
    # test model ability with loaded model
    agent.load(f"{logdir / 'common_model.pth'}")
    agent.epsilon = 0
    loaded_score = np.mean([test_with_seed(env, agent, seed) for seed in seeds])
    print('loaded_score is :', loaded_score)

    return best_score


def test_with_seed(env, agent, seed):
    cur_state = env.reset(seed=seed)
    rates = []
    while True:
        action = agent.get_action(cur_state)[0]
        next_state, reward, done, info = env.step(
            action.astype(np.int32))
        rates.append(reward)
        cur_state = next_state.reshape((-1, env.n_states))
        if done:
            break
    # print(f'seed {seed}, score {np.mean(rates)}', flush=True)
    return np.mean(rates)


def get_dqn_agent(env, **kwargs):
    n_states = env.n_states
    n_actions = env.n_actions
    agent = DQN(n_states, n_actions, **kwargs)
    return agent


def get_instances(args=utils.get_args()):
    env = utils.get_env(**args.env)
    agent = get_dqn_agent(env, **args.agent)
    conf = utils.get_config('config.yaml')
    conf['env'].update(args.env)
    conf['agent'].update(args.agent)
    logdir = utils.get_logdir(conf)
    return env, agent, logdir


def demo(env, agent, logdir):
    dqn_result = dqn_loop(env, agent, logdir)

    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        # RL results
        f.write('dqn: ' + str(dqn_result) + '\r\n')
        # benckmarks, use default seed
        env.reset(seed=980615)
        results = cal_benchmarks(env)
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
    print('done')


if __name__ == '__main__':
    instances = get_instances()
    demo(*instances)
