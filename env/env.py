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
        return self.map.get_state()

    def step(self, actions):
        code = []
        for i in range(len(actions)):
            cs = self.map.execute(i, actions[i])
            code = code + cs

        reward1 = 0
        for c in code:
            if c == data.map.COMMON:
                reward1 = reward1 - 1
            elif c == data.map.HIT_TEAM:
                reward1 = reward1 - 200
            elif c == data.map.HIT_WORLD:
                reward1 = reward1 - 200
            elif c == data.map.HIT_BARRIER:
                reward1 = reward1 - 200
            elif c == data.map.SCORE:
                reward1 = reward1 + 500
            elif c == data.map.INVALID:
                continue

        if len(self.map.camps2) == 0:
            return self.map.get_state(), reward1, True

        count = 0
        c1 = self.map.camps1
        c2 = self.map.camps2
        dis = numpy.zeros((len(c1), len(c2)))
        for i in range(len(c1)):
            for j in range(len(c2)):
                dis[i][j] = max(abs(c1[i].x - c2[j].x), abs(c1[i].y - c2[j].y))

        for i in range(len(c2)):
            if min(dis[:][i]) > 1:
                count = count + 1

        return self.map.get_state(), reward1 - count, False


if __name__ == "__main__":
    env = Env()
    env.map.out()
