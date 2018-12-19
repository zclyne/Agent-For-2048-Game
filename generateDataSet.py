GAME_SIZE = 4
SCORE_TO_WIN = 2048

from game2048.game import Game
from game2048.agents import ExpectiMaxAgent

# save the dataset
f_256 = open("dataset_256.txt", "w")
f_512 = open("dataset_512.txt", "w")
f_1024 = open("dataset_1024.txt", "w")

for i in range(30000):
    print("i = ", i)
    game = Game(size=GAME_SIZE)
    agent = ExpectiMaxAgent(game=game)
    while True:
        direction = agent.step()
        if (game.end == True):
            break
        maxNum = 0
        for i in range(4):
            for j in range(4):
                if game.board[i, j] > maxNum:
                    maxNum = game.board[i, j]
        if maxNum == 2048: # start the next turn
            break
        if maxNum <= 256:
            for i in range(4):
                for j in range(4):
                    print(game.board[i, j], file = f_256)
            print(direction, file = f_256)
        elif maxNum == 512:
            for i in range(4):
                for j in range(4):
                    print(game.board[i, j], file = f_512)
            print(direction, file = f_512)
        if maxNum == 1024:
            for i in range(4):
                for j in range(4):
                    print(game.board[i, j], file = f_1024)
            print(direction, file = f_1024)
        game.move(direction)