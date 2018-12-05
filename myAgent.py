import numpy as np
from keras.models import load_model

class Agent:
    '''Agent Base.'''

    def __init__(self, game, display=None):
        self.game = game
        self.display = display

    def play(self, max_iter=np.inf, verbose=False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    def step(self):
        direction = int(input("0: left, 1: down, 2: right, 3: up = ")) % 4
        return direction

class myAgent(Agent):

    def __init__(self, game, display=None):
        if game.size != 4:
            raise ValueError(
                "`%s` can only work with game of `size` 4." % self.__class__.__name__)
        super().__init__(game, display)
        self.game = game
        self.model = load_model('myAgent.h5')

    def step(self):
        inputboard = np.zeros((1, 16, 16, 1))
        for i in range(16):
            num = self.game.board[i // 4, i % 4]
            if num == 0:
                inputboard[0, i, 0, 0] = 1
            else:
                inputboard[0, i, int(np.log2(num)), 0] = 1
        direction = self.model.predict(inputboard).tolist()[0]
        return direction.index(max(direction))