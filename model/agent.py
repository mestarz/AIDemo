"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import numpy as np
import random

import configure
from brain import Brain
from memory.memory import Memory

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0


class Agent(object):
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, state_size, action_size, bee_index, brain_name, test=False):
        self.state_size = state_size
        self.action_size = action_size
        self.bee_index = bee_index
        self.learning_rate = configure.LEARNING_RATE
        self.gamma = 0.95
        self.brain = Brain(self.state_size, self.action_size, brain_name, test)
        self.memory = Memory(configure.MEMORY_CAPACITY)
        self.update_target_frequency = configure.TARGET_FREQUENCY
        self.max_exploration_step = configure.MAXIMUM_EXPLORATION
        self.batch_size = configure.BATCH_SIZE
        self.step = 0
        self.test = test
        if self.test:
            self.epsilon = MIN_EPSILON

    def greedy_actor(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            return np.argmax(self.brain.predict_one_sample(state))

    def find_targets_uer(self, batch):
        batch_len = len(batch)

        states = np.array([o[0] for o in batch])
        states_ = np.array([o[3] for o in batch])

        p = self.brain.predict(states)
        p_target_ = self.brain.predict(states_, target=True)

        x = np.zeros((batch_len, self.state_size))
        y = np.zeros((batch_len, self.action_size))
        errors = np.zeros(batch_len)

        for i in range(batch_len):
            o = batch[i]
            s = o[0]
            a = o[1][self.bee_index]
            r = o[2]
            done = o[4]

            t = p[i]
            old_value = t[a]
            if done:
                t[a] = r
            else:
                t[a] = r + self.gamma * np.amax(p_target_[i])

            x[i] = s
            y[i] = t
            errors[i] = np.abs(t[a] - old_value)

        return [x, y]

    def observe(self, sample):
        self.memory.remember(sample)

    def decay_epsilon(self):
        # slowly decrease Epsilon based on our experience
        self.step += 1

        if self.test:
            self.epsilon = MIN_EPSILON
            self.beta = MAX_BETA
        else:
            if self.step < self.max_exploration_step:
                self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * (
                        self.max_exploration_step - self.step) / self.max_exploration_step
                self.beta = MAX_BETA + (MIN_BETA - MAX_BETA) * (
                        self.max_exploration_step - self.step) / self.max_exploration_step
            else:
                self.epsilon = MIN_EPSILON

    def replay(self):
        batch = self.memory.sample(self.batch_size)
        x, y = self.find_targets_uer(batch)
        self.brain.train(x, y)

    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.brain.update_target_model()
