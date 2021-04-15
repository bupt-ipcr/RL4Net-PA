import utils
import numpy as np
from pa_dqn import DQN
from torch.utils.tensorboard import SummaryWriter
from benckmarks import cal_benchmarks
MAX_EPISODES = 1000
DECAY_THRES = 300


def rl_loop(env, agent, logdir):
    summary_writer = SummaryWriter(log_dir=logdir)
    # train
    train_his = []
    for ep in range(MAX_EPISODES):
        cur_state = env.reset()
        cur_state = cur_state.reshape((-1, env.n_states))
        done = False
        ep_his = []
        agent.epsilon = max((DECAY_THRES - ep) / DECAY_THRES, 0.001)
        while True:
            action = agent.get_action(cur_state)[0]
            next_state, reward, done, info = env.step(action.astype(np.int32))
            next_state = next_state.reshape((-1, env.n_states))
            agent.add_steps(cur_state, action, reward, done, next_state)
            loss = agent.learn()
            if loss:
                summary_writer.add_scalar('loss', loss, agent.eval_step)
            cur_state = next_state
            ep_his.append(reward/env.n_recvs)
            if done:
                cum_reward = np.mean(ep_his)
                summary_writer.add_scalar('reward', cum_reward, ep)
                train_his.append({'cum_reward': cum_reward, 'ep_his': ep_his})
                if len(train_his) % 10 == 0:
                    print('EP: ', len(train_his),  'DQN:',
                          np.mean([t['cum_reward'] for t in  train_his[-10:]]), flush=True)
                break
    print('calculating benckmarks')
    # find best ep_his
    train_his.sort(key=lambda o: o['cum_reward'], reverse=True)
    dqn_result = train_his[0]['cum_reward'], train_his[0]['ep_his']
    results = cal_benchmarks(env)
    result_path = logdir / 'results.log'
    with result_path.open('w') as f:
        for result in results:
            f.write(result[0] + ': ' + str(result[1]) + '\r\n')
        f.write('dqn: ' + str(dqn_result[0]) + '\r\n')
        f.write(str(dqn_result[1]))
    print('done')


def get_instance():
    env = utils.get_env()
    # for k, v in env.__dict__.items():
    #     print(f'{k}: {v}')
    n_states = env.n_states
    n_actions = env.n_actions
    agent = DQN(n_states, n_actions)
    logdir = utils.get_logdir()
    return env, agent, logdir


if __name__ == '__main__':
    from datetime import datetime
    start = datetime.now()
    instances = get_instance()
    rl_loop(*instances)
    end = datetime.now()
    print('cost time:', end - start)
