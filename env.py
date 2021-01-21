import numpy as np



class GNN_env(object):
    def __init__(self, action_value, subgraph_num, initial_k):
        super(GNN_env, self).__init__()
        self.action_space = ['add', 'minus']
        self.action_value = action_value
        self.n_actions = len(self.action_space)
        self.n_features = subgraph_num
        self.k = initial_k

    def reset(self):
        initial_observation = self.get_observation()
        initial_observation = np.array(initial_observation)
        return initial_observation

    def get_observation(self):
        select_num = int(self.k * self.n_features)
        observation = select_num
        return observation

    def step(self, action, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, last_acc):
        if action == 0:  # add
            if self.k <= (1 - self.action_value):
                self.k += self.action_value
        elif action == 1:  # minus
            if self.k >= self.action_value:
                self.k -= self.action_value

        eva_loss, eva_acc, _, rl_reward, _, _, _, _ = net.evaluate(test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, self.k)
        s_ = self.get_observation()
        # print(s_)
        s_ = np.array(s_)
        # reward function
        if rl_reward > last_acc:
            reward = 1
            done = True
        elif rl_reward < last_acc:
            reward = -1
            done = True
        else:
            reward = 0
            done = True
        return s_, reward, done, rl_reward
