from keras.layers.core import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from source.individual.brain.brain_settings import HIDDEN_LAYERS_QUANTITY, NEURONS_QUANTITY, DATASET_SIZE, EPOCHS, \
    BATCH_SIZE

from random import randint
import matplotlib.pyplot as plt
import numpy as np
import os


## Policy network model class
class PolicyNetwork:
    def __init__(self):  # Later - more parameters
        self.history = None

        self.checkpoint_path = "networks/network/trainings/training_2/cp.ckpt"
        self.checkpoint_abs_path = os.path.abspath(self.checkpoint_path)

        self.layers_quant = HIDDEN_LAYERS_QUANTITY
        self.neurons_quant = NEURONS_QUANTITY

        self.model = Sequential()

        # Input layer
        self.model.add(Dense(2, input_dim=2))

        # Hidden layers
        for i in range(HIDDEN_LAYERS_QUANTITY):
            self.model.add(Dense(NEURONS_QUANTITY, activation='relu'))
            self.model.add(Dropout(0.2))

        # Output layer
        self.model.add(Dense(1, activation='relu'))  # from -12 to 9

        # Compile model
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    ## Function that creates full dataset with fixed size
    # @input The size of needed dataset
    # @return Number pairs array and true value of their sum array
    def create_full_dataset(self, size=DATASET_SIZE):

        training_set = []
        true_results_set = []

        for _ in range(size):
            train, result = self.create_train()
            training_set.append(train)
            true_results_set.append(result)

        numpy_training_set = np.array(training_set)
        numpy_true_results_set = np.array(true_results_set)

        return numpy_training_set, numpy_true_results_set

    ## Function that creates one random game position
    # @return  Number pair and true value of their sum
    def create_train(self):
        number1 = randint(0, 100)
        number2 = randint(0, 100)
        sum = number1 + number2
        return np.array([number1, number2]), sum

    ## Function that starts network training
    def start_training(self):
        checkpoint_callback = ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1)
        training_set, true_results_set = self.create_full_dataset()
        # true_results_set = true_results_set.reshape(10000, 1, 1)
        self.history = self.model.fit(training_set, true_results_set, epochs=EPOCHS,
                                      batch_size=BATCH_SIZE,
                                      callbacks=[checkpoint_callback])

    ## A function that visualizes the results of the last training session in the form of graphs of changes in
    # accuracy and losses over time
    def visualize_studying_results(self):
        # print(self.history.history.keys())
        # summarize history for accuracy
        plt.plot(self.history.history['accuracy'])
        # plt.plot(self.history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # self.history.history[]
        # summarize history for loss
        plt.plot(self.history.history['loss'])
        # plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    ## A function that evaluates the policy for the transferred game state
    # @param hand: Cards in players hand
    # @param board: Cards on board (the unknown are coded as 00)
    # @return The policy value of the current state
    def predict(self, number1, number2):
        input = np.array([number1, number2])
        return self.model.predict(input)

    ## A function that evaluates the current version of network
    # @return Average accuracy and loss for a random test data set
    def evaluate(self):
        x_test, y_test = self.create_full_dataset(1000)
        value = self.model.evaluate(x_test, y_test, 100)
        return value

    ## Function loading the weights of the latest trained version of the neural network
    # @param path: The path to the weight data directory
    def load(self, path=None):
        if path == None:
            path = self.checkpoint_path
        self.model.load_weights(path)
