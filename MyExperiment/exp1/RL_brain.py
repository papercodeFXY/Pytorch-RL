import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import pandas as pd
import random
from cluster_env import Cluster
import datetime
import matplotlib.pyplot as plt
import pylab as pl
from itertools import chain

def state_init():
    init_state = pd.DataFrame(np.zeros(12*4).reshape(12, 4), columns=[0, 1, 2, 3])
    for i in range(len(init_state)):
        j = random.randint(0, 3)
        init_state.iloc[i][j] = 1
    return init_state

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY = 2000
server_attribute = pd.DataFrame(np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                                          0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
                                          1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
                                          0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]).
                                reshape(4, 12),
                                columns=np.arange(12))
init_state = state_init()
env = Cluster(init_state, server_attribute)
N_ACTIONS = len(env.action_space)
N_STATES = 48
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        #self.out = nn.Linear(50, N_ACTIONS)
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    curr_time1 = datetime.datetime.now()
    improve_list = []
    dqn = DQN()

    print('\nCollecting experience...')
    cost_all_list = []
    reward_all_list = []
    init_reward = env.reward(env.cost_all(env.cost_init), env.state_init)
    for i_episode in range(60000):
        epoch_curr_time1 = datetime.datetime.now()
        # initial state
        state_init_arr = env.state_array(env.state_init)
        state = (env.state_init).copy()
        costs = env.cost_init
        sum = 0
        reward_list = [0]
        state_arr_for_one = state_init_arr
        reward = init_reward
        ep_r = 0
        while True:
            action = dqn.choose_action(list(chain.from_iterable(np.array(state))))
            # take action
            state_, costs_, reward_, cost_all, is_better = env.step(action, state, costs)

            state__arr = env.state_array(state_)

            different = [y for y in (state_arr_for_one + state__arr) if y not in state_arr_for_one]
            print("different:", different)
            if ((reward_ < init_reward and reward_ < min(reward_list) or
                 (len(different) == 0 and reward_ >= reward and reward_ > (init_reward)))):
                done = True
            else:
                done = False
            # RL learn from this transition
            print("done:", done)

            reward = reward_
            reward_list.append(reward)
            # # modify the reward
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            # r = r1 + r2
            dqn.store_transition(list(chain.from_iterable(np.array(state))), action, reward, list(chain.from_iterable(np.array(state_))))

            ep_r += reward
            if dqn.memory_counter > MEMORY_CAPACITY:
                dqn.learn()
                if done:
                    print('Ep: ', i_episode,
                          '| Ep_r: ', round(ep_r, 2))

            sum += 1

            if done:
                break

            state_arr_for_one = state__arr
            different_init = [y for y in (state_init_arr + state__arr) if y not in state_init_arr]

            costs = costs_
            state = state_

        reward_all_list.append(reward)
        epoch_curr_time2 = datetime.datetime.now()
        epoch_time = epoch_curr_time2 - epoch_curr_time1
        # if (action in actions and q_table.loc[str(state), action] >= 0) and (done and q_table.loc[str(state), action] >= 0 and reward > 0):
        #     break
        # else:
        #     actions.append(action)
        # break while loop when end of this episode

        # if done and q_table.loc[str(state),action]!=0:
        #     break

        cost_all_list.append(cost_all)
        print("epoch:", i_episode+1)
        print("The number of cycles in this epoch：", sum)
        print("The reward list:", reward_list)
        print("The best reward in this epoch：", max(reward_list))
        print("The final reward in this epoch:", reward)
        print("The final cost in this epoch:", cost_all)
        print("当前状态与初始状态的差别", different_init)
        print("当前状态与初始状态的差别数", len(different_init))
        print("epoch_time:", epoch_time, "\n")

    print("------------------------")
    print("The final state_array:", state__arr)
    print("The final cost:", cost_all)

    improve = ((reward_all_list[-1] - init_reward)/init_reward)*100
    print("The improve percent:", improve, "%")

    improve_list.append(improve)

    y_1 = reward_all_list
    y_all_list = y_1
    x = (np.arange(len(y_all_list)))
    y = y_all_list
    y1 = [init_reward]*len(x)
    fig = plt.Figure(figsize=(14, 10))
    pl.plot(x, y, label=u'RL')
    pl.legend()
    pl.plot(x, y1, label=u'Init')
    pl.legend()
    pl.xlabel(u"epoch", size=14)
    pl.ylabel(u"reward", size=14)
    plt.show()

    curr_time2 = datetime.datetime.now()
    train_time = curr_time2-curr_time1
    print("The training time：", train_time)