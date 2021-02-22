'''
Это моя игра змейка
Которую я разрабатываю 2021-02-22
'''

import gym
import gym_snake

import numpy as np

from collections import deque

from snake_dqn import DQN, init_state

from snake_learn import q_learning

x_size = 6
y_size = 6

ACTIONS = [0, 1, 2, 3]

#DQN param
n_state = x_size * y_size
n_hidden = [108, 12]
lr = 0.003
n_action = len(ACTIONS)
n_episode = 1000
replay_size = 32
target_update = 10

## Creating environment
env = gym.make('snake-v0')
env.unit_size = 1
env.unit_gap = 0
env.snake_size = 2
env.grid_size = [x_size,y_size]

## Observing snake for now
obs = env.reset()

init_state(env.controller)

obs = env.reset()

dqn = DQN(n_state, n_action, n_hidden, lr)

memory = deque(maxlen=10000)

total_reward_episode = [0] * n_episode
q_learning(env, dqn, n_episode, ACTIONS, total_reward_episode, memory, replay_size, target_update, gamma=.9, epsilon=1)
    
env.close()

print("Done")