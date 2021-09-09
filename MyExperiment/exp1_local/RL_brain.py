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

# 按照概率生成要移动到的服务器
def generate_server(list_var_small_all, probabilities_list):

    if not (0.85 < sum(probabilities_list) < 1.15):
        raise ValueError("The probabilities are not normalized!")
    if len(list_var_small_all) != len(probabilities_list):
        raise ValueError("The length of two input lists are not match!")

    random_normalized_num = random.random()  # random() -> x in the interval [0, 1).
    accumulated_probability = 0.0
    for item in zip(list_var_small_all, probabilities_list):
        accumulated_probability += item[1]
        if random_normalized_num < accumulated_probability:
            return item[0]

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
        init_reward, init_list_var, load_weight_var = env.reward(env.cost_all(env.cost_init), env.state_init)
        init_reward_list.append(init_reward)
        print("init_reward:", init_reward)
        print("init_list_var", init_list_var)
        print("初始方差", load_weight_var)

    costs = env.cost_init
    state = (env.state_init).copy()

    # 对QSs中的数据进行遍历，按照本地化的原则在属性所在的服务器中选择一个reward最大的服务器
    for item in QSs:
        action = []
        action.append(item[0])
        reward_list = []
        for i in item[1]:
            action.append(i)
            state_, costs_, reward_, cost_all, list_var, load_weight_var = env.step(action, state, costs)
            reward_list.append(reward_)
        max_index = reward_list.index(max(reward_list))
        action.append(item[1][max_index])
        state_, costs_, reward_, cost_all, list_var, load_weight_var = env.step(action, state, costs)
        state = state_
        costs = costs_
    print("未考虑负载均衡时的方差", load_weight_var)
    print("未考虑负载均衡时用于求方差的数组", list_var)
    print("reward_:", reward_)

    for n in range(3):
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
        QSs = [QSs[i] for i in range(len(QSs)) if i not in index]
        # 把出现数量多的属性组合对应的查询分配给目前数量少的服务器
        list_var_mean = sum(list_var)/len(list_var)
        for item in QSs_mode:
            list_var_small_all = []
            list_var_small_value = []
            for server, value in enumerate(list_var):
                list_var_small = []
                if value < list_var_mean:
                    list_var_small.append(server)
                    list_var_small.append(value)
                    list_var_small_value.append(1/value)
                    list_var_small_all.append(list_var_small)
            print("list_var_small_all", list_var_small_all)
            action = []
            action.append(item[0])
            probabilities_list = []
            for i in list_var_small_value:
                probabilities = i/sum(list_var_small_value)
                probabilities_list.append(probabilities)
            print(probabilities_list)
            list_var_small_generate = generate_server(list_var_small_all, probabilities_list)
            server = int(list_var_small_generate[0])
            action.append(server)
            print("action", action)
            state_, costs_, reward_, cost_all, list_var, load_weight_var = env.step(action, state, costs)
            state = state_
            costs = costs_
        print("方差_%d: %f" % (n, load_weight_var))
        print("用于求方差的数组_%d: %s" % (n, str(list_var)))
        print("reward_:", reward_)



    improve = ((reward_ - (sum(init_reward_list)/len(init_reward_list)))/(sum(init_reward_list)/len(init_reward_list)))*100
    print("The improve percent:", improve, "%")

    curr_time2 = datetime.datetime.now()
    calculate_time = curr_time2-curr_time1
    print("The calculate time：", calculate_time)