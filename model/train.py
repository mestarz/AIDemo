"""
Created on Wednesday Jan  16 2019

@author: Seyed Mohammad Asghari
@github: https://github.com/s3yyy3d-m
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

import configure
from env.env import Env
from model.agent import Agent

ARG_LIST = ['learning_rate', 'optimizer', 'memory_capacity', 'batch_size', 'target_frequency', 'maximum_exploration',
            'max_timestep', 'first_step_memory', 'replay_steps', 'number_nodes', 'target_type', 'memory',
            'prioritization_scale', 'dueling', 'agents_number', 'grid_size', 'game_mode', 'reward_mode']


def get_name_brain(idx):
    return './result/weights/' + '_' + str(idx) + '.h5'


def get_name_rewards():
    return './result/rewards/' + '.csv'


def get_name_timesteps():
    return './result/timesteps/' + '.csv'


class Environment(object):

    def __init__(self, test=False):
        self.env = Env()
        self.episodes_number = configure.EPISODE_NUMBER
        self.max_ts = configure.MAX_TIMESTEP
        self.test = test
        self.filling_steps = configure.FIRST_STEP_MEMORY
        self.steps_b_updates = configure.REPLAY_STEP

        self.num_agents = configure.AGENTS_NUMBER
        self.num_landmarks = self.num_agents

    def run(self, agents, file1, file2):

        total_step = 0
        rewards_list = []
        timesteps_list = []
        max_score = -10000
        for episode_num in range(self.episodes_number):
            state = self.env.reset()

            # converting list of positions to an array
            state = np.array(state)
            state = state.ravel()

            done = False
            reward_all = 0
            time_step = 0
            while not done and time_step < self.max_ts:

                if time_step % 10 == 0:
                    env.env.map.tracks.clear()

                actions = []
                for agent in agents:
                    actions.append(agent.greedy_actor(state))
                next_state, reward, done = self.env.step(actions)
                # converting list of positions to an array
                next_state = np.array(next_state)
                next_state = next_state.ravel()

                if not self.test:
                    for agent in agents:
                        agent.observe((state, actions, reward, next_state, done))
                        if total_step >= self.filling_steps:
                            agent.decay_epsilon()
                            if time_step % self.steps_b_updates == 0:
                                agent.replay()
                            agent.update_target_model()

                total_step += 1
                time_step += 1
                state = next_state
                reward_all += reward

            rewards_list.append(reward_all)
            timesteps_list.append(time_step)

            print("Episode {p}, Score: {s}, Final Step: {t}, Goal: {g}".format(p=episode_num, s=reward_all,
                                                                               t=time_step, g=done))

            if not self.test:
                if episode_num % 100 == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(file1)

                    df = pd.DataFrame(timesteps_list, columns=['steps'])
                    df.to_csv(file2)

                    for agent in agents:
                        agent.brain.save_model(str(episode_num))

                    # if total_step >= self.filling_steps:
                    #     if reward_all > max_score:
                    #         for agent in agents:
                    #             agent.brain.save_model()
                    #         max_score = reward_all


if __name__ == "__main__":
    # DQN Parameters
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

    env = Environment()
    state_size = env.env.state_size
    action_space = env.env.action_space

    all_agents = []
    for b_idx in range(configure.AGENTS_NUMBER):
        brain_file = get_name_brain(b_idx)
        all_agents.append(Agent(state_size, action_space, b_idx, brain_file))

    rewards_file = get_name_rewards()
    timesteps_file = get_name_timesteps()

    env.run(all_agents, rewards_file, timesteps_file)
