
import numpy as np
import pygame


class MemNN:
    def __init__(self, layers=[8,4,2], steps=3):
        self.layers = layers
        self.n_layers = 1

        self.steps = steps

        self.Z = [None] * (len(self.layers)-1)
        self.A = [None] * (len(self.layers) - 1)

        self.total_size = sum(layers)

        #self.learning_rate = 1
        self.mutation_rate = 0.5
        self.mutation_chance = 0.4

    def forward_prop(self, input):

        self.A[0][:self.layers[0]] = input
        for i in range(self.steps):
            self.Z[0] = self.W[0].dot(self.A[0]) + self.b[0]
            self.A[0] = np.concatenate(ReLU(self.Z[:-self.layers[-1]]), self.Z[-self.layers[-1]:])

        acceleration = self.A[-1][0, 0]
        steering = self.A[-1][1, 0]
        return acceleration, steering
        return self.Z[-1] # maybe should add sigmoid


    def mutate(self, bestnn):
        for j in range(len(self.W)):
            self.W[j] = bestnn.W[j] + self.__get_random_variation(self.W[j].shape[0], self.W[j].shape[1])
            self.b[j] = bestnn.b[j] + self.__get_random_variation(self.b[j].shape[0], self.b[j].shape[1]) * 0.2
            
            self.W[j] = np.clip(self.W[j], -1, 1)
            self.b[j] = np.clip(self.b[j], -1, 1)
    
    def __get_random_variation(self, x, y):
        mask = np.random.rand(x, y) < self.mutation_chance # only mutate 10% of the weights
        return self.mutation_rate * (np.random.rand(x,y) * 2 - 1) * mask
        
    def init(self):
        self.W = [np.random.rand(self.layers[i+1], self.layers[i]) - 0.5 for i in range(len(self.layers)-1)]
        self.b = [np.random.rand(self.layers[i+1], 1) - 0.5 for i in range(len(self.layers)-1)]
        return self
    
    def copy_weights(self, other):
        self.layers = other.layers
        self.W = [np.copy(w) for w in other.W]
        self.b = [np.copy(b) for b in other.b]
    
    def save(self, path):
        data = {'W':self.W, 'b':self.b}
        np.save("saved/" + path, data)
        print('Model saved !')

    def load(self, path):
        data = np.load("saved/" + path + ".npy", allow_pickle=True).item()
        self.W = data['W']
        self.b = data['b']
        print('Model loaded ! :', path)





def ReLU(x):
    return np.maximum(0, x)

def ReLU_der(x):
    return x > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))