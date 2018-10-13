import sys
import pickle
import matplotlib.pyplot as plt
import sneks
import numpy
from sneks.core.world import Snek
import numpy as np
import gym
import random
from time import sleep
import operator
import pprint, pickle

# pkl_file = open('q_table_test1_50000_v3_final.pkl', 'rb')
#
# q_table = pickle.load(pkl_file)


class Main():
    def __init__(self):
        self.env = gym.make('snek-v1')
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space
        self.total_episodes = 200000
        self.learning_rate = 0.2
        self.gamma = 0.4
        self.max_steps = 1000
        self.epsilon = 0.2
        self.max_epsilon = 1.0
        self.min_epsilon = 0.1
        self.decay_rate = 0.0001
        self.d = {}
        self.d_reward = {}
        self.q_table = {}
        self.valid_dir = [0, 1, 2, 3]
        self.directions = {'left': 3, 'right': 1, 'up': 0, 'down': 2}
        self.flag = 1

    def is_wall_near_by(self,snake_position):
        left, right, up, down = False, False, False, False
        block_size = 15
        if snake_position[1] - 1 < 0:
            left = True
        if snake_position[1] + 1 > 15:
            right = True
        if snake_position[0] - 1 < 0:
            up = True
        if snake_position[0] + 1 > 15:
            down = True
        temp = (left, right, up, down)
        return temp
    
    def Q_learning(self):
        for episode in range(self.total_episodes):
            self.state = self.env.reset()
            r = 0
            count = 1
            for (x, y), value in np.ndenumerate(self.state):
                if self.state[x, y] == 101.0:
                    self.snake_position = (x, y)
                if self.state[x, y] == 100.0:
                    self.initial_tail = (x, y)
                if self.state[x, y] == 255.0:
                    self.apple_position = (x, y)

            self.wall_info = tuple(self.is_wall_near_by(self.snake_position))
            self.cur_state = self.snake_position + self.apple_position + self.initial_tail + self.wall_info

            if self.cur_state not in self.q_table:
                self.flag = 1
                self.q_table[self.cur_state] = {val: 0 for val in self.valid_dir}

            done = False
            for step in range(self.max_steps):

                if self.flag == 1:
                    self.action = self.env.action_space.sample()  # Exploration
                    self.flag = 0
                else:
                    if numpy.random.random() > self.epsilon:
                        if len(set(self.q_table[self.cur_state].values())) == 1:
                            self.action = self.env.action_space.sample()
                        else:
                            self.action = max(self.q_table[self.cur_state].items(), key=operator.itemgetter(1))[0]

                    else:
                        self.action = self.env.action_space.sample()

                list = self.env.step(self.action)

                self.new_state, self.reward, self.done, self.info, self.tail_gate, self.head = list[0], list[1], list[2], list[3], list[4], list[5]
                r += self.reward

                for (x, y), value in np.ndenumerate(self.new_state):
                    if self.new_state[x, y] == 255.0:
                        apple_new_position = (x, y)

                wall_info = tuple(self.is_wall_near_by(self.head))
                cur_new_state = self.tail_gate[0] + self.apple_new_position + self.tail_gate[-1] + wall_info

                if cur_new_state not in self.q_table:
                    self.flag = 1
                    self.q_table[cur_new_state] = {val: 0 for val in self.valid_dir}

                self.env.render()
                self.next_max = max(self.q_table[self.cur_new_state].items(), key=operator.itemgetter(1))[0]
                self.q_table[self.cur_state][self.action] += self.learning_rate * (self.reward + self.gamma * self.next_max - self.q_table[self.cur_state][self.action])

                self.state = self.new_state
                self.cur_state = self.cur_new_state

                if done == True:
                    break

            self.d_reward[episode + 1] = r
            self.epsilon = self.min_epsilon + (self.max_epsilon-self.min_epsilon) * np.exp(-self.decay_rate*self.episode)
            output = open('q_table_Q_test1_50000_v3_final.pkl', 'wb')
            pickle.dump(self.q_table, self.output)


            output.close()
            output1 = open('q_table_Q_test1_50000_v4_final_reward.pkl', 'wb')
            pickle.dump(self.d_reward, self.output1)

            output1.close()

            self.d_reward_update = {}
            temp = 0
            for k, v in dict.items(self.d_reward):
                temp += v
                if k % 500 == 0:
                    self.d_reward_update[k] = temp / 500
                    temp = 0

            p = []
            q = []
            for k, v in dict.items(self.d_reward_update):
                p.append(k)
                q.append(v)

            plt.plot(p, q, color="blue")
            plt.xlabel('Number of Episodes')
            plt.ylabel('Average Rewards')
            plt.title('Number of Episodes vs Average Rewards')
            plt.legend(loc='best')
            plt.show()
            self.env.close()
            sys.exit(0)


if __name__ == '__main__':
    Main()




