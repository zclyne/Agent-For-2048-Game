GAME_SIZE = 4
SCORE_TO_WIN = 2048

from game2048.game import Game
from game2048.agents import ExpectiMaxAgent

# save the dataset
f = open("dataset_2.txt", "w")

for i in range(300):
    print("i = ", i)
    game = Game(size=GAME_SIZE)
    agent = ExpectiMaxAgent(game=game)
    while True:
        direction = agent.step()
        if (game.end == True):
            break
#        print (game.board)
#        print ("direction: ", direction)
        for i in range(4):
            for j in range(4):
                print(game.board[i, j], file = f)
        print(direction, file = f)
        game.move(direction)