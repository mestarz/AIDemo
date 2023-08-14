import numpy as np

import data.soldier
from data.soldier import Soldier

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

# Response
COMMON = 0
HIT_TEAM = 1
HIT_BARRIER = 2
HIT_WORLD = 3
SCORE = 4
INVALID = 5

# Kind
NORMAL = 0
CAMP1 = 1
CAMP2 = 2
BARRIER = 3
TRACK = 4
EMPTY = 5

A_DIFF = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


def inbound(x, y, size):
    return 0 <= x < size and 0 <= y < size


class Map(object):

    def __init__(self):
        self.size = configure.GRID_SIZE
        self.map = np.zeros((self.size, self.size), dtype=int)
        self.max_barrier = int(self.size * self.size * 0.2)
        self.soldier_nums = configure.AGENTS_NUMBER
        self.size = configure.GRID_SIZE
        self.state_size = self.size * self.size * 5
        self.action_space = len(A_DIFF)
        self.barriers = []
        self.camps1 = []
        self.camps2 = []
        self.tracks = []

        self.reset()

    def reset(self):
        data.soldier.COUNT_ID = 0
        self.map = np.zeros((self.size, self.size), dtype=int)
        self.tracks.clear()
        self.random_init()
        self.update()

    # action [ cid, k, act ]
    # 标号，第k步，动作
    def execute(self, action):
        code = 0
        c = self.camps1[action[0]]

        if c is None:
            return code

        nx = c.x + A_DIFF[action[2]][0]
        ny = c.y + A_DIFF[action[2]][1]

        if not inbound(nx, ny, self.size) or self.map[nx][ny] != NORMAL:
            nx = c.x
            ny = c.y

        if nx != c.x or ny != c.y:
            self.tracks.append((nx, ny))
            self.map[c.x][c.y] = TRACK
            self.map[nx][ny] = CAMP1
            c.x = nx
            c.y = ny

        for c2 in self.camps2:
            if abs(c2.x - c.x) <= 1 and abs(c2.y - c.y) <= 1 and action[1] == configure.STEP:
                c2.life = c2.life - c.atk
                if c2.life <= 0:
                    self.camps2.remove(c2)
                    self.map[c2.x][c2.y] = NORMAL
                code = code + 1

        return code

    def find(self, cid):
        for c in self.camps1:
            if c.id == cid:
                return c
        return None

    # [cx, cy], [tx, ty], k, [size, size], [周围障碍], [视野地图]
    # 2 + 2 + 1 + 9 + view * view * 4
    def get_state(self):
        a = np.array(self.map, dtype=int)
        cate = 5
        one_hot_encoded_map = np.zeros((a.shape[0], a.shape[1], cate), dtype=int)

        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                category = a[i, j]
                one_hot_encoded_map[i, j, category] = 1
        return one_hot_encoded_map

    def random_init(self):
        cells = [(i, j) for i in range(0, self.size) for j in range(0, self.size)]
        positions_idx = np.random.choice(len(cells), size=self.soldier_nums + self.soldier_nums + self.max_barrier,
                                         replace=False)

        c1 = positions_idx[0:self.soldier_nums]
        self.camps1 = [Soldier(cells[c][0], cells[c][1]) for c in c1]
        c2 = positions_idx[self.soldier_nums:self.soldier_nums * 2]
        self.camps2 = [Soldier(cells[c][0], cells[c][1]) for c in c2]

        self.barriers = [cells[p] for p in positions_idx[self.soldier_nums * 2:]]

    def update(self):
        self.map = np.zeros((self.size, self.size))

        for c in self.camps1:
            self.map[c.x][c.y] = CAMP1
        for c in self.camps2:
            self.map[c.x][c.y] = CAMP2
        for p in self.barriers:
            self.map[p[0]][p[1]] = BARRIER
        for p in self.tracks:
            self.map[p[0]][p[1]] = TRACK

    def out(self):
        for i in range(self.size):
            for j in range(self.size):
                k = self.map[i][j]
                if k == NORMAL:
                    print("  ", end="")
                elif k == BARRIER:
                    print("M ", end="")
                elif k == CAMP1:
                    print("1 ", end="")
                elif k == CAMP2:
                    print("2 ", end="")
                elif k == TRACK:
                    print("+ ", end="")
            print()
        print("----------------------------------")
        print()
