from RL_brain import DQN
from cluster_env import Cluster


if __name__ == '__main__':
    dqn = DQN()

    print('\nCollecting experience...')
    for i_episode in range(400):
        s = env.reset()
        ep_r = 0
        while True:
            env.render()
            a = dqn.choose_action(s)

            # take action
            s_, r, done, info = env.step(a)

            # modify the reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            r = r1 + r2

            dqn.store_transition(s, a, r, s_)

            ep_r += r
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            if done:
                break
            s = s_