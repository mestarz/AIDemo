import numpy as np

import configure

# action
STOP = 0
UP = 1
UP_RIGHT = 2
RIGHT = 3
DOWN_RIGHT = 4
DOWN = 5
DOWN_LEFT = 6
LEFT = 7
LEFT_UP = 8

KIND_NUMS = 5
# Kind
NORMAL = 0
BARRIER = 1
SOLIDER = 2
FLAG = 3
TRACE = 4

A_DIFF = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def inbound(x, y, size):
    return 0 <= x < size and 0 <= y < size


class Env(object):

    def __init__(self):
        self.seed = None
        self.size = configure.GRID_SIZE
        self.map = np.zeros((self.size, self.size), dtype=int)
        self.max_barrier = int(self.size * self.size * 0.2)
        self.flag_nums = configure.FLAG_NUMS
        self.action_space = len(A_DIFF)

        self.soldier = None
        self.flags = []
        self.barriers = []
        self.trace = []

        self.reset()

    def reset(self):
        self.random_init()
        self.update()

    def seed_reset(self):
        self.seed_init(self.seed)
        self.update()

    def step(self, action):
        if action == 0:  # STOP
            return -10, self.get_state()

        new_position = tuple(np.array(A_DIFF[action]) + np.array(self.soldier))
        if not inbound(new_position[0], new_position[1], self.size):
            return -100, self.get_state()
        elif self.map[new_position] == TRACE:
            return -100, self.get_state()
        elif self.map[new_position] == BARRIER:
            return -100, self.get_state()
        elif self.map[new_position] == FLAG:
            return -100, self.get_state()
        else:
            self.trace.append(self.soldier)
            self.soldier = new_position
            self.map[self.soldier] = TRACE

            count = 0
            for f in self.flags:
                if max(abs(f - np.array(new_position))) <= 1:
                    count += 1
                    self.flags.remove(f)

            return 1 + count * 500, self.get_state()

    def execute(self, action):
        trace = []
        code = 0
        position = np.array(self.soldier)
        trace.append(tuple(position))
        for a in action:
            if a == 0:  # STOP
                code -= 10
                continue
            new_position = tuple(np.array(A_DIFF[a]) + position)
            if not inbound(new_position[0], new_position[1], self.size):
                code -= 100
                continue
            elif new_position in trace:
                code -= 10
                continue
            elif self.map[new_position] == NORMAL:
                trace.append(new_position)
                code += 1
                position = new_position
                continue
            elif self.map[new_position] == BARRIER:
                code -= 100
                continue
            elif self.map[new_position] == FLAG:
                code -= 10
                continue

        self.trace = trace

        count = 0
        for f in self.flags:
            for t in trace:
                if max(abs(f - np.array(t))) <= 1:
                    count += 1
                    break

        return code + count * 500, count == self.flags

    def get_state(self):
        return np.eye(KIND_NUMS, dtype=int)[np.array(self.map.copy(), dtype=int)]

    def random_init(self):
        cells = [(i, j) for i in range(0, self.size) for j in range(0, self.size)]

        positions_idx = np.random.choice(len(cells), size=(1 + self.flag_nums) * configure.CONFUSE + self.max_barrier,
                                         replace=False)

        self.seed = positions_idx
        self.seed_init(positions_idx)

    def seed_init(self, seed):
        self.flags.clear()
        self.barriers.clear()

        cells = [(i, j) for i in range(0, self.size) for j in range(0, self.size)]

        np.random.shuffle(seed)
        self.soldier = cells[seed[0]]
        self.flags = [cells[i] for i in seed[configure.CONFUSE: configure.CONFUSE + self.flag_nums]]
        self.barriers = [cells[i] for i in seed[(1 + self.flag_nums) * configure.CONFUSE:]]

    def update(self):
        self.map[:] = 0

        self.map[self.soldier] = SOLIDER
        for f in self.flags:
            self.map[f] = FLAG
        for b in self.barriers:
            self.map[b] = BARRIER

    def out(self):
        for i in range(self.size):
            for j in range(self.size):
                if self.map[i, j] == SOLIDER:
                    print("X ", end="")
                    continue
                elif tuple((i, j)) in self.trace:
                    print("+ ", end="")
                    continue

                k = self.map[i][j]
                if k == NORMAL:
                    print("  ", end="")
                elif k == BARRIER:
                    print("M ", end="")
                elif k == FLAG:
                    print("$ ", end="")
            print()
        print("----------------------------------")
        print()


if __name__ == "__main__":
    env = Env()
    env.reset()
    steps = (1, 2, 4, 5, 2, 3, 1, 4, 8, 2)
    codes = []
    for s in steps:
        code = env.step(s)
        codes.append(code)
    print(env.trace)
    print(codes)
    state = env.get_state()
    print(state)
    env.out()
