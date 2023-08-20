import os
import random
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

import configure
from model.agent import Agent
from model.env import Env


def get_name_brain():
    return "brain.h5"


def get_name_rewards():
    return 'reward.csv'


class Environment(object):

    def __init__(self):
        self.env = Env()
        self.episodes_number = configure.EPISODE_NUMBER
        self.max_ts = configure.MAX_TIMESTEP
        self.filling_steps = configure.FIRST_STEP_MEMORY
        self.steps_b_updates = configure.REPLAY_STEP

        self.num_agents = configure.AGENTS_NUMBER
        self.num_landmarks = self.num_agents

    def run(self, brain, reward_file):

        rewards_list = []
        current_episode = 0
        pbar = tqdm(initial=current_episode, total=self.episodes_number, unit='episodes')
        while current_episode < self.episodes_number:

            self.env.reset()
            state = self.env.get_state()
            for _ in range(configure.ONE_MAP_TRAIN_NUM):

                if np.random.rand() < brain.epsilon:
                    if np.random.rand() < 0.8:
                        actions = self.env.get_guidance()
                    else:
                        actions = [random.randint(0, brain.action_size - 1) for _ in range(configure.STEP)]
                else:
                    actions = agent.predict(state)

                reward, done = self.env.execute(actions)
                brain.observe(state, actions, reward)
                if current_episode % self.steps_b_updates == 0:
                    brain.train()

                current_episode += 1
                agent.decay_epsilon()

                pbar.update(1)
                print("Episode {p}, Score: {s}".format(p=current_episode, s=reward))
                rewards_list.append(reward)

                if current_episode % 1000 == 0:
                    df = pd.DataFrame(rewards_list, columns=['score'])
                    df.to_csv(reward_file)
                if current_episode % 200000 == 0:
                    brain.save_model(str(current_episode))


if __name__ == "__main__":
    # DQN Parameters
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0: INFO, 1: WARNING, 2: ERROR, 3: FATAL

    stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    env = Environment()
    action_space = env.env.action_space

    brain_file = get_name_brain()
    agent = Agent(weight_file="brain.h5")
    rewards_file = get_name_rewards()

    env.run(agent, rewards_file)
