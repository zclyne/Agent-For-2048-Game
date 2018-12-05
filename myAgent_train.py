import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout

# define constants
NUM_EPOCHS = 2
NUM_CLASSES = 4 # four directions
BATCH_SIZE = 1000
INPUT_SHAPE = (16, 16, 1)

# open file
f = open("dataset_3000.txt")

# initialize neural network
model = Sequential()
model.add(Conv2D(512, (2, 2), strides=(2, 2), activation='relu', input_shape=INPUT_SHAPE, padding='valid'))
model.add(Conv2D(256, (4, 1), strides=(1, 1), activation='relu'))
model.add(Conv2D(256, (1, 4), strides=(1, 1), activation='relu'))
model.add(Conv2D(128, (2, 1), strides=(1, 1), activation='relu'))
model.add(Conv2D(128, (1, 2), strides=(1, 1), activation='relu'))
model.add(Flatten())
model.add(Dense(2000, activation='relu'))
model.add(Dense(500, activation='relu'))
#model.add(Dense(200, activation='relu'))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(16, activation='relu'))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

# compile the network
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print("Compile Complete")

for k in range(10):
    # save boards and directions
    boards = []
    directions = []
    
    # read data
    for j in range(1000000):
        num = f.readline()
        if not num: # end of file
            break
        num = float(num)
        # the 4 * 4 board is changed into a 16 * 16 one hot encoding board
        # every row corresponds to one number in the original board
        # num is represented by setting board[i, int(log2(num))] to 1
        # as there are 16 columns in every block, the maximum number that can be represented is 32768
        # since there is no 1s in the board, we can use board[i, 0] = 1 to represent units that are empty
        board = np.zeros((16, 16, 1)) 
        for i in range(16):
            if num == 0:
                board[i, 0, 0] = 1
            else:
                board[i, int(np.log2(num)), 0] = 1
            num = float(f.readline())
    #    board = np.zeros((16, 1))
    #    for i in range(16):
    #        board[i, 0] = num;
    #        num = float(f.readline())
#        board = np.zeros((4, 4, 1))
#        for i in range(4):
#            for j in range(4):
#                board[i, j, 0] = num;
#                num = float(f.readline())
        boards.append(board) # save the board
        direction = int(num)
        directions.append(direction) # save the direction
    # convert to numpy array
    boards = np.array(boards)
    directions= np.array(directions)
    # convert to one-hot encoding
    directions = keras.utils.to_categorical(directions, num_classes=NUM_CLASSES)
    # train
    model.fit(boards, directions, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)
    
# save the model
model.save('myAgent.h5')