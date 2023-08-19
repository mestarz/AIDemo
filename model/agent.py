"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""
from collections import deque

import random
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten
from keras import optimizers

import configure
from model.env import A_DIFF, KIND_NUMS, Env

MAX_EPSILON = 1.0
MIN_EPSILON = 0.01

MIN_BETA = 0.4
MAX_BETA = 1.0

HUBER_LOSS_DELTA = 2.0


@tf.function
def huber_loss(y_true, y_predict):
    err = y_true - y_predict

    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)

    return K.mean(loss)


class Agent(object):
    epsilon = MAX_EPSILON
    beta = MIN_BETA

    def __init__(self, weight_file=None, test=False):
        self.test = test
        self.action_size = len(A_DIFF)
        self.learning_rate = configure.LEARNING_RATE
        self.update_target_frequency = configure.TARGET_FREQUENCY
        self.gamma = 0.95

        self.memory = deque(maxlen=configure.MEMORY_CAPACITY)
        self.max_exploration_step = configure.MAXIMUM_EXPLORATION
        self.batch_size = configure.BATCH_SIZE

        self.weight_backup = weight_file

        self.model = self._build_model_cnn()
        self.target_model = self._build_model_cnn()
        self.step = 0
        if self.test:
            self.epsilon = MIN_EPSILON

    def _build_model_cnn(self):
        model = Sequential()
        model.add(Input(shape=(configure.GRID_SIZE, configure.GRID_SIZE, KIND_NUMS)))
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2, padding='same'))
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2, padding='same'))
        model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2, padding='same'))
        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=2, padding='same'))
        model.add(Flatten())
        model.add(Dense(units=4096, activation='relu'))
        model.add(Dense(units=4096, activation='relu'))
        model.add(Dense(units=1024, activation='relu'))
        model.add(Dense(units=configure.STEP * self.action_size, activation='linear'))

        opter = optimizers.rmsprop_v2.RMSprop(learning_rate=configure.LEARNING_RATE)
        # opter = optimizers.adam_v2.Adam(learning_rate=configure.LEARNING_RATE)
        model.compile(loss='mse', optimizer=opter)
        if self.test:
            if not os.path.isfile(self.weight_backup):
                print('Error:no file')
            else:
                model.load_weights(self.weight_backup)

        return model

    def _build_mode_rnn(self):
        pass

    def save_model(self, prefix):
        self.model.save(prefix + self.weight_backup)

    def predict_actions(self, input_state):
        if np.random.rand() < self.epsilon:
            return [random.randint(0, 8) for _ in range(configure.STEP)]
        else:
            res = self.model.predict(input_state).flatten()
            return [np.argmax(res[i * self.action_size:(i + 1) * self.action_size]) for i in range(configure.STEP)]

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

    def observe(self, current_state, action, reward):
        self.memory.append((current_state, action, reward))

    def train(self):
        if len(self.memory) < configure.FIRST_STEP_MEMORY:
            return

        samples = random.sample(self.memory, self.batch_size)
        current_input = np.stack([sample[0] for sample in samples])
        current_q_value = self.model.predict(current_input)

        for i, (current_state, action, reward) in enumerate(samples):
            out_action = [i * self.action_size + act for i, act in enumerate(action)]
            current_q_value[i, out_action] = reward

        hist = self.model.fit(current_input, current_q_value, batch_size=configure.BATCH_SIZE, verbose=0, shuffle=False)
        loss = hist.history['loss'][0]
        return loss

    def update_target_model(self):
        if self.step % self.update_target_frequency == 0:
            self.target_model.set_weights(self.model.get_weights())


if __name__ == "__main__":
    a = Agent("20000brain.h5")
    env = Env()
    state = env.get_state()
    for _ in range(3):
        actions = a.predict_actions(state)
        code, done = env.execute(actions)
        env.out()
        print(actions, code)

    # a.train()
