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

DataSet_filePath = "QueryAttribute_poission_50"
server_number = 8
server_attribute = pd.DataFrame(np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,
                                          1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                                          0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                                          0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]).
                                reshape(8, 24),
                                columns=np.arange(24))

def read_file(DataSet_filePath):
    with open(DataSet_filePath, 'r') as f:
        content = f.readlines()
        QSs = []
        for item in content:
            QS = []
            item = item.strip("\n")
            q = item.split(",")[0]
            targetAttribute = item.split(",")[1:]
            targetAttribute = list(map(int, targetAttribute))
            servers = []
            for attribute in targetAttribute:
                server = server_attribute[server_attribute.loc[:, attribute] == 1].index[0]
                servers.append(server)
            QS.append(int(q))
            QS.append(servers)
            QSs.append(QS)
    return QSs

def state_init():
    init_state = pd.DataFrame(np.zeros(len(read_file(DataSet_filePath))*server_number).reshape(len(read_file(DataSet_filePath)), server_number), columns=np.arange(server_number))
    for i in range(len(init_state)):
        j = random.randint(0, len(init_state.columns)-1)
        init_state.iloc[i][j] = 1
    return init_state

# Hyper Parameters
BATCH_SIZE = 256
LR = 0.0005                   # learning rate
EPSILON = 0.8               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 5    # target update frequency
MEMORY_CAPACITY = 10000
# server_attribute = pd.DataFrame(np.array([0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
#                                           0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
#                                           1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
#                                           0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1]).
#                                 reshape(4, 12),
#                                 columns=np.arange(12))
init_state = state_init()
QSs = read_file(DataSet_filePath)
env = Cluster(init_state, server_attribute, QSs, server_number)
N_ACTIONS = len(env.action_space)
N_STATES = len(env.state_init)*len(env.state_init.columns)
# ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape
device = torch.device('cuda:0'if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        #self.out = nn.Linear(50, N_ACTIONS)
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            is_random = False
            # action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            is_random = True
            # action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action, is_random

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
            self.target_net.to(device)
            # print(next(self.target_net.parameters()).device)
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # print("loss:", loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    curr_time1 = datetime.datetime.now()
    dqn = DQN()
    print('\nCollecting experience...')
    cost_all_list = []
    reward_all_list = []
    init_reward = env.reward(env.cost_all(env.cost_init), env.state_init)
    for i_episode in range(60000):
        print("数据集是泊松分布,reward是两个tanh")
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
            state_list = list(chain.from_iterable(np.array(state)))
            action, is_random = dqn.choose_action(state_list)
            # take action
            state_, costs_, reward_, cost_all = env.step(action, state, costs)
            state__arr = env.state_array(state_)
            different = [y for y in (state_arr_for_one + state__arr) if y not in state_arr_for_one]
            # print("different:", different)
            if ((reward_ < init_reward and reward_ < min(reward_list)) or
                 (not is_random and len(different) == 0 and reward_ >= reward and reward_ > (init_reward))):
                done = True
            else:
                done = False

            if reward_ < init_reward and reward_ < min(reward_list):
                print("负结束")

            if not is_random and len(different) == 0 and reward_ >= reward and reward_ > (init_reward):
                print("正结束")

            reward = reward_
            reward_list.append(reward)
            dqn.store_transition(state_list, action, reward, list(chain.from_iterable(np.array(state_))))
            ep_r += reward
            if dqn.memory_counter > MEMORY_CAPACITY:
                # print("learn")
                dqn.learn()
            sum += 1

            if done:
                break

            state_arr_for_one = state__arr
            costs = costs_
            state = state_
        reward_all_list.append(reward)
        epoch_curr_time2 = datetime.datetime.now()
        epoch_time = epoch_curr_time2 - epoch_curr_time1
        cost_all_list.append(cost_all)
        print("epoch:", i_episode+1)
        print("The number of cycles in this epoch：", sum)
        # print("The best reward in this epoch：", max(reward_list))
        print("The final reward in this epoch:", reward)


    print("------------------------")
    improve = ((reward_all_list[-1] - init_reward)/init_reward)*100
    print("The improve percent:", improve, "%")

    reward_all_list_np = np.array(reward_all_list)
    y_all_list = np.mean(reward_all_list_np.reshape(-1, 100), axis=1)
    print("y_all_list:", len(y_all_list))
    x = (np.arange(len(y_all_list)))
    y = y_all_list
    y1 = [init_reward]*len(x)
    fig = plt.Figure(figsize=(24, 20))
    pl.plot(x, y, label=u'RL')
    pl.legend()
    pl.plot(x, y1, label=u'Init')
    pl.legend()
    pl.xlabel(u"epoch", size=14)
    pl.ylabel(u"reward", size=14)
    plt.savefig("D:/SynologyDrive/Paper/BIP/Experiment/实验结果_查询数50/mac-test.png")
    plt.show()
    curr_time2 = datetime.datetime.now()
    train_time = curr_time2-curr_time1
    print("The training time：", train_time)