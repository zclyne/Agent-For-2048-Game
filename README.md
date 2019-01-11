# 2048Game
This is a course project for SJTU EE369, an agent for 2048 web app. Used neural network as my agent.  
For boards with the highest score less than or equal to 256, use model_256.  
For boards with the highest score equal to 512, use model_512.  
For boards with the highest score equal to 1024, use model_1024.  
This agent achieved an average score of 1300.48 in 50 games, each game ends when the highest score in the board is 2048.  
# File structure and how to use this agent:  
1. Execute generateDataSet.py to generate offline dataset that is divided into three files according to the highest score.  
2. Execute myAgent_256_train.py, myAgent_512_train.py and myAgent_1024_train.py to define the neural network and train it on the dataset.  
3. Execute myAgent_train_online.py to train each network online.
4. myAgent.step() get the input of the 4 * 4 2048 game board and gives a direction(0, 1, 2 and 3 correspond to left, down, right and up respectively) as the output.  
5. To play 2048 with my agent in linux, download webapp.py and go to my another repository "2048-api" and download /game2048 and /static. After all these are done, type "python webapp.py" in the terminal where you save these files to start playing.
