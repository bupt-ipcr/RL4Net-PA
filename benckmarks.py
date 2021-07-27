
import utils
import numpy as np

def mock_FP_algorithm(env):
    P = np.random.rand(env.n_recvs)
    fading = env.fading
    g_ii = fading.diagonal()
    for _ in range(100):
        P_last = P
        recv_power = P * fading
        signal_power = recv_power.diagonal()
        total_power = recv_power.sum(axis=1)
        inter_noise = total_power - signal_power + \
            1e-3*pow(10., env.thres_power/10.)

        gamma = signal_power / inter_noise
        y = np.sqrt((1.+gamma) * signal_power) / inter_noise
        y_j = np.tile(np.expand_dims(y, axis=1), [1, env.n_recvs])
        P = np.minimum(env.power_levels[-1], np.square(y) *
                       (1.+gamma) * g_ii / np.sum(np.square(y_j)*fading, axis=1))
        if np.linalg.norm(P_last - P) < 1e-3:
            break
    return P


def mock_WMMSE_algorithm(env):
    v = np.random.rand(env.n_recvs)  # max_power*np.ones((N))
    fading = env.fading
    recv_power = np.square(v) * fading
    signal_power = recv_power.diagonal()
    total_power = recv_power.sum(axis=1)
    inter_noise = total_power - signal_power + \
        1e-3*pow(10., env.thres_power/10.)

    u = np.sqrt(signal_power) / total_power
    w = 1 + signal_power / inter_noise
    C = np.sum(w)
    for _ in range(100):
        C_last = C
        W = w * np.ones([env.n_recvs, env.n_recvs])
        U = u * np.ones([env.n_recvs, env.n_recvs])
        v_ = W*U**2*fading
        v = w*u*np.sqrt(fading.diagonal()) / np.sum(v_, axis=1)
        v = np.minimum(
            np.sqrt(env.power_levels[-1]), np.maximum(1e-10*np.random.rand(env.n_recvs), v))

        recv_power = np.square(v) * fading
        signal_power = recv_power.diagonal()
        total_power = recv_power.sum(axis=1)
        inter_noise = total_power - signal_power + \
            1e-3*pow(10., env.thres_power/10.)

        u = np.sqrt(signal_power) / total_power
        w = 1 + signal_power / inter_noise
        C = np.sum(w)
        if np.abs(C_last - C) < 1e-3:
            break
    P = v**2
    return P


def random_algorithm(env):
    n_actions = env.n_actions
    return np.random.randint(0, n_actions, env.n_t * env.m_r)


def maximum_algorithm(env):
    n_actions = env.n_actions
    return np.full(env.n_t * env.m_r, n_actions - 1)


def cal_benchmark(algorithm, env):
    env.reset()
    cum_r = []
    while True:
        p = algorithm.func(env)
        raw = algorithm.type == 'power'
        s_, r, d, i = env.step(p, raw=raw)
        cum_r.append(r)
        if d:
            return algorithm.name, np.mean(cum_r)


def cal_benchmarks(env, algs=None):
    from collections import namedtuple
    Algorithm = namedtuple('Algorithm', 'name func type')
    algorithms = [
        Algorithm('fp', mock_FP_algorithm, 'power'),
        Algorithm('wmmse', mock_WMMSE_algorithm, 'power'),
        Algorithm('random', random_algorithm, 'action'),
        Algorithm('maximum', maximum_algorithm, 'action')
    ]
    
    if algs:
        algorithms = filter(algorithms, lambda alg: alg.name in algs)

    results = [cal_benchmark(algorithm, env) for algorithm in algorithms]
    return results


if __name__ == '__main__':
    args = utils.get_args_from_config()
    env = args.Env(**args.env)
    results = cal_benchmarks(env)
    for name, reward in results:
        print(f'{name}: {reward}')
