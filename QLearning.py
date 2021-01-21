import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.85, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_, done):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        # if s_ != 'terminal':
        #     q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        # else:
        #     q_target = r  # next state is terminal
        if not done:
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )



def isTerminal(k_record, limited_epochs, delta_k, start_check_epochs=300)->bool:
    """
   After $start_check_epochs, check the $range of k's change in last $limited_epochs.
   if the $range less than $delta_k,

   *for example k_record = [ 0.1, 0.2 ,0.3, 0.3] the last 3 change is [0.2, 0.3, 0.3],
   the range of it is 0.1 (0.3 - 0.2), which less than 0.3(if the delta k is 0.3)
    """
    assert start_check_epochs > limited_epochs, 'the length of k_record is not long enough'
    if len(k_record) >= start_check_epochs:
        record_len = len(k_record)
        record_last_limited_epochs = np.array(k_record[record_len-limited_epochs:])
        range_ = np.max(record_last_limited_epochs) - np.min(record_last_limited_epochs)

        if range_ <= delta_k:
            return True
        else:
            return False

    else:
        return False




def run_QL(env, RL, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, initial_acc):
    observation = env.reset()
    while True:
        action = RL.choose_action(str(observation))
        observation_, reward, done, initial_acc = env.step(action, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, initial_acc)

        RL.learn(str(observation), action, reward, str(observation_), done)
        env.k = round(env.k, 4)
        if done:
            return env.k, reward


def generate_experience(env, RL, net, test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, initial_acc):
    observation = env.reset()
    RL.check_state_exist(str(observation))
    initial_k = round(env.k, 4)
    k_list = np.linspace(0, 1, 21)
    k_list = k_list[1:]
    for k in k_list:
        env.k = round(k, 4)
        observation = env.reset()
        RL.check_state_exist(str(observation))
        action1 = RL.actions[0]
        observation_, reward, done = env.step(action1, net, test_x, test_dsi, test_sadj, test_t, test_t_mi,
                                              test_mask, initial_acc)
        RL.learn(str(observation), action1, reward, str(observation_), done)
        # print(k, 'action1', action1, observation, observation_, reward)
        RL.learn(str(observation), action1, reward, str(observation_), done)

        env.k = round(k, 4)
        observation = env.reset()
        RL.check_state_exist(str(observation))
        action2 = RL.actions[1]
        observation_, reward, done = env.step(action2, net, test_x, test_dsi, test_sadj, test_t, test_t_mi,
                                              test_mask, initial_acc)
        RL.learn(str(observation), action1, reward, str(observation_), done)
        # print(k, 'action2', action2, observation, observation_, reward)
    env.k = initial_k


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
