import numpy as np
import pandas as pd
import random
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


class Cluster(tk.Tk, object):
    def __init__(self, state_init, server_attribute, QSs, server_number):
        super(Cluster, self).__init__()
        self.server_number = server_number
        self.cost_matrix = pd.DataFrame(np.array([[0,1,5,12,7,10,15,9],
                                                  [1,0,4,2,8,6,11,10],
                                                  [5,4,0,3,11,13,8,5],
                                                  [12,2,3,0,7,6,10,4],
                                                  [7,8,11,7,0,12,9,5],
                                                  [10,6,13,6,12,0,3,8],
                                                  [15,11,8,10,9,3,0,10],
                                                  [9,10,5,4,5,8,10,0]]),
                                        columns=[0, 1, 2, 3, 4, 5, 6, 7])
        self.server_attribute = server_attribute
        self.QSs = QSs
        self.action_space = np.array(self.action_list())
        self.n_actions = len(self.action_space)
        self.state_init = state_init

        self.cost_init = self.cost_init()


    def step(self, action, state, costs):
        s = state.copy()
        #action_real[查询，移动到的服务器]
        if isinstance(action, np.ndarray):
           action = action[0]
        action_real = self.action_space[action]
        q = action_real[0]
        index_server = action_real[1]
        s.iloc[q, :] = 0
        s.iloc[q, index_server] = 1

        cost_new = self.cost_caculate(q, index_server)

        costs[q] = cost_new
        cost_all = self.cost_all(costs)
        reward = self.reward(cost_all, s)
        s_ = s

        return s_, costs, reward, cost_all


    # generate the total action set
    def action_list(self):
        action = []
        for i in range(len(self.QSs)):
            for j in range(self.server_number):
                li = []
                li.append(i)
                li.append(j)
                action.append(li)
        return action


    # compute the initial costs array based on the initial state matrix. every element represent the total cost of the query
    def cost_init(self):
        state_init = self.state_init
        # print(len(state_init))
        states = self.state_array(state_init)
        # print(len(states))
        costs = []
        # print(len(state_init))
        for i in range(len(state_init)):
            index_server = states[i][1]
            cost = self.cost_caculate(i,  index_server)
            costs.append(cost)
        return costs


    def cost_caculate(self,q,index_server):
        cost = 0
        for j in range(len(self.QSs[q][1])):
            target_server = self.QSs[q][1][j]
            cost += self.cost_matrix.iloc[index_server, target_server]
        return cost


    # compute the total reward based on the costs array
    def cost_all(self, costs):
        cost_all = 0
        for i in range(len(costs)):
            cost_all += costs[i]
        return cost_all

    def reward(self, cost_all, state):
        list = []
        for i in state.columns:
            list.append(state[i].sum())

        load_weight_var = np.var(list)
        # reward = 100*self.tanh((10*len(state)/cost_all))*self.tanh(10*self.function(1.1, load_weight_var))
        reward = 100*10*(len(state)/cost_all)*self.tanh(100*self.function(1.1, load_weight_var))
        # reward = (len(state)/cost_all) * 100 * self.function(1.1, load_weight_var)
        return reward

    def function(self, a, x):
        y = 1/(a**x)
        return y

    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    # transform the state matrix into array
    def state_array(self, state):
        states = []
        for i in range(len(state)):
            for j in range(len(state.columns)):
                state_arr = []
                if state.iloc[i, j] == 1:
                    state_arr.append(i)
                    state_arr.append(j)
                    states.append(state_arr)
        return states
