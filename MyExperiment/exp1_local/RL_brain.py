import numpy as np
import gym
import pandas as pd
import random
from cluster_env import Cluster
import datetime

DataSet_filePath = "./QueryAttribute_longtail"
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
    QSs = read_file(DataSet_filePath)
    for i in range(episode):
        init_state = state_init()
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
    # 将数据集中的众数（也就是出现次数最多的元素）去除，以免影响方差
    attri = []
    for item in QSs:
        attri.append(item[1])
    x = dict((tuple(a), attri.count(a)) for a in attri)
    y = [k for k, v in x.items() if max(x.values()) == v]
    y = [list(item) for item in y]
    index = []
    for j in y:
        for i, item in enumerate(attri):
            if item == j:
                index.append(i)
    print("index的数量", len(index))

    QSs_mode = [QSs[i] for i in range(len(QSs)) if i in index]
    QSs_after_delete = [QSs[i] for i in range(len(QSs)) if i not in index]
    # 对QSs中的数据进行遍历，按照本地化的原则在属性所在的服务器中选择一个reward最大的服务器
    for item in QSs_after_delete:
        action = []
        action.append(item[0])
        reward_list = []
        for i in item[1]:
            action.append(i)
            state_, costs_, reward_, cost_all, list_var = env.step(action, state, costs)
            reward_list.append(reward_)
        max_index = reward_list.index(max(reward_list))
        action.append(item[1][max_index])
        state_, costs_, reward_, cost_all = env.step(action, state, costs)
        state = state_
        costs = costs_

    # 把出现数量多的属性组合对应的查询分配给目前数量少的服务器
    list_var_mean = sum(list_var)/len(list_var)
    list_var_small_all = []
    list_var_small = []
    for value in list_var:
        if value < list_var_mean:
            for server, item in enumerate(list_var):
                if item == value:
                    list_var_small.append(server)
                    list_var_small.append(value)
                list_var_small_all.append(list_var)






    print("reward_:", reward_)

    improve = ((reward_ - (sum(init_reward_list)/len(init_reward_list)))/(sum(init_reward_list)/len(init_reward_list)))*100
    print("The improve percent:", improve, "%")

    curr_time2 = datetime.datetime.now()
    caculate_time = curr_time2-curr_time1
    print("The caculate time：", caculate_time)