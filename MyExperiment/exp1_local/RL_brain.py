import numpy as np
import gym
import pandas as pd
import random
from cluster_env import Cluster
import datetime

DataSet_filePath = "./QueryAttribute_1"
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
            # print(targetAttribute)
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



if __name__ == '__main__':
    episode = 100
    for i in range(episode):
        init_state = state_init()
        QSs = read_file(DataSet_filePath)
        env = Cluster(init_state, server_attribute, QSs, server_number)
        # print("cost_all", env.cost_all((env.cost_init)))
        N_ACTIONS = len(env.action_space)
        N_STATES = len(env.state_init)*len(env.state_init.columns)
        curr_time1 = datetime.datetime.now()
        init_reward_list = []
        init_reward = env.reward(env.cost_all(env.cost_init), env.state_init)
        init_reward_list.append(init_reward)
        print("init_reward:", init_reward)
    costs = env.cost_init
    state = (env.state_init).copy()
    reward__list = []
    QSs.reverse()
    for item in QSs:
        action = []
        action.append(item[0])
        reward_list = []
        for i in item[1]:
            action.append(i)
            state_, costs_, reward_, cost_all = env.step(action, state, costs)
            reward_list.append(reward_)
        max_index = reward_list.index(max(reward_list))
        action.append(item[1][max_index])
        state_, costs_, reward_, cost_all = env.step(action, state, costs)
        state = state_
        costs = costs_

    print("reward_:", reward_)

    improve = ((reward_ - (sum(init_reward_list)/len(init_reward_list)))/(sum(init_reward_list)/len(init_reward_list)))*100
    print("The improve percent:", improve, "%")

    curr_time2 = datetime.datetime.now()
    caculate_time = curr_time2-curr_time1
    print("The caculate time：", caculate_time)