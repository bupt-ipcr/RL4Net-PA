#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@author: Jiawei Wu
@create time: 1970-01-01 08:00
@edit time: 2021-04-19 10:22
@file: /RL4Net-PA/pa_dqn.py
@desc: 
"""
import utils
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks
MAX_EPISODES = 1000
DECAY_THRES = 700


@utils.timeit
def dqn_loop(env, agent, logdir):
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    print(f"Start DQN loop.")
    train_his = []
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((-1, env.n_states))
        done = False
        ep_his = []
        agent.epsilon = max((DECAY_THRES - ep) / DECAY_THRES, 0.001)
        while True:
            action = agent.get_action(cur_state)[0]
            next_state, reward, done, info = env.step(
                action.astype(np.int32))
            next_state = next_state.reshape((-1, env.n_states))
            agent.add_steps(cur_state, action, reward, done, next_state)
            loss = agent.learn()
            if loss:
                summary_writer.add_scalar('loss', loss, agent.eval_step)
            cur_state = next_state
            ep_his.append(reward)
            if done:
                cum_reward = np.mean(ep_his)
                summary_writer.add_scalar('reward', cum_reward, ep)
                train_his.append({'cum_reward': cum_reward, 'ep_his': ep_his})
                if len(train_his) % 10 == 0:
                    print('EP: ', len(train_his),  'DQN:',
                          np.mean([t['cum_reward'] for t in train_his[-10:]]), info, flush=True)
                break
    agent.save(episode=MAX_EPISODES)
    # find best ep_his
    train_his.sort(key=lambda o: o['cum_reward'], reverse=True)
    dqn_result = train_his[0]['cum_reward'], train_his[0]['ep_his']
    return dqn_result


def get_instances(args):
    Env, DQN = args.Env, args.DQN
    env = Env(**args.env)
    agent = DQN(env.n_states, env.n_actions, **args.agent)
    logdir = args.logdir
    if not logdir.exists():
        logdir.mkdir(parents=True)
    return env, agent, logdir


def demo(env, agent, logdir):
    dqn_result = dqn_loop(env, agent, logdir)

    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        # RL results
        f.write('dqn: ' + str(dqn_result[0]) + '\r\n')
        # f.write(str(dqn_result[1]))
        # benckmarks
        results = cal_benchmarks(env)
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
    print('done')


if __name__ == '__main__':
    args = utils.get_args_from_config('config.yaml')
    instances = get_instances(args)
    demo(*instances)
