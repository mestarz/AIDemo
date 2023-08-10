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

A_DIFF = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]


class Map(object):

    def __init__(self):
        self.size = configure.GRID_SIZE
        self.map = np.zeros((self.size, self.size))
        self.max_barrier = int(self.size * self.size * 0.2)
        self.soldier_nums = configure.AGENTS_NUMBER
        self.max_tracks = 10 * self.soldier_nums
        self.state_size = (self.soldier_nums * 2 + self.max_tracks + self.max_barrier) * 2
        self.action_space = len(A_DIFF)
        self.barriers = []
        self.camps1 = []
        self.camps2 = []
        self.tracks = []

        self.reset()

    def reset(self):
        data.soldier.COUNT_ID = 0
        self.map = np.zeros((self.size, self.size))
        self.tracks.clear()
        self.random_init()
        self.update()

    def execute(self, cid, action):
        code = []
        c = self.find(cid)

        if c is None:
            return [INVALID]

        nx = c.x + A_DIFF[action][0]
        ny = c.y + A_DIFF[action][1]

        if action == STOP:
            code.append(COMMON)
        else:
            if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
                code.append(HIT_WORLD)
            else:
                s = self.map[nx][ny]
                if s == CAMP1 or s == TRACK:
                    code.append(HIT_TEAM)
                elif s == BARRIER:
                    code.append(HIT_BARRIER)
                elif s == CAMP2:
                    code.append(SCORE)

        if len(code) > 0:
            nx = c.x
            ny = c.y

        if nx != c.x or ny != c.y:
            self.map[c.x][c.y] = NORMAL
            self.map[nx][ny] = CAMP1
            c.x = nx
            c.y = ny

        need_update = False
        for c2 in self.camps2:
            if abs(c2.x - c.x) <= 1 and abs(c2.y - c.y) <= 1:
                c2.life = c2.life - c.atk
                if c2.life <= 0:
                    need_update = True
                code.append(SCORE)

        if need_update:
            self.update()

        if len(code) == 0:
            code.append(COMMON)

        return code

    def find(self, cid):
        for c in self.camps1:
            if c.id == cid:
                return c
        return None

    def get_state(self):
        state = []

        def col_camp(cs, max_len):
            for c in cs:
                state.append(c.x)
                state.append(c.y)
            max_len = max_len - len(cs)
            for i in range(max_len):
                state.append(-1)
                state.append(-1)

        col_camp(self.camps1, self.soldier_nums)
        col_camp(self.camps2, self.soldier_nums)

        def col_poi(ps, max_len):
            for p in ps:
                state.append(p[0])
                state.append(p[1])
            max_len = max_len - len(ps)
            for i in range(max_len):
                state.append(-1)
                state.append(-1)

        col_poi(self.barriers, self.max_barrier)
        col_poi(self.tracks, self.max_tracks)

        return state

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
        for c in self.camps1:
            if c.life <= 0:
                self.camps1.remove(c)
        for c in self.camps2:
            if c.life <= 0:
                self.camps2.remove(c)

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
