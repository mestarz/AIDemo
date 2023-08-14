import numpy

import data.soldier
from data.map import Map


class Env(object):

    def __init__(self):
        self.map = Map()
        self.state_size = self.map.state_size
        self.action_space = self.map.action_space

    def reset(self):
        self.map.reset()

    # len(actions) == 5
    # action [ cid, k, act ]
    # 标号，第k步，动作
    def step(self, actions):
        code = 0
        for i in range(len(actions)):
            code = code + self.map.execute((0, i + 1, actions[i])) * 500

        if len(self.map.camps2) == 0:
            return self.map.get_state(), code, True

        return self.map.get_state(), code, False


if __name__ == "__main__":
    env = Env()
    env.map.out()
