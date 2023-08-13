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

    # action [ cid, k, act ]
    # 标号，第k步，动作
    def step(self, action):
        code = self.map.execute(action) * 50

        if len(self.map.camps2) == 0:
            return self.map.get_state(action[0]), code, True

        count = 0
        c1 = self.map.camps1[action[0]]
        for c in self.map.camps2:
            if abs(c.x - c1.x) > 1 or abs(c.y - c1.y) > 1:
                count = count + 1

        return self.map.get_state(action[0]), code - count, False


if __name__ == "__main__":
    env = Env()
    env.map.out()
