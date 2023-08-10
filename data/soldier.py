COUNT_ID = 0


class Soldier(object):

    def __init__(self, x, y):
        global COUNT_ID
        self.life = 100  # 生命值
        self.atk = 50  # 攻击力
        self.id = COUNT_ID  # 初始id
        COUNT_ID = COUNT_ID + 1
        self.x = x
        self.y = y
