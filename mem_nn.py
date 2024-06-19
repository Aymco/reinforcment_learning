
import numpy as np
import pygame


class MemNN:
    def __init__(self, layers=[8,4,2], steps=1):
        self.total_size = sum(layers)
        self.out_size = self.total_size - layers[0]
        self.layers = layers

        self.steps = steps


        self.values = np.zeros((self.out_size, 1))


        #self.learning_rate = 1
        self.mutation_rate = 0.5
        self.mutation_chance = 0.4

    def forward_prop(self, input):
        #self.A[0][:self.layers[0]] = input[:, 0]
        #for i in range(self.steps):
        #self.Z[0] = self.W[0].dot(self.A[0]) + self.b[0]
        #self.Z[0][:-self.layers[-1]] = ReLU(self.Z[0][:-self.layers[-1]])
        #acceleration = self.Z[0][-2, 0]
        #steering = self.Z[0][-1, 0]
        # values is  a column vector
        inp_and_values = np.concatenate((input, self.values), axis=0)
        self.values = self.W[0].dot(inp_and_values) + self.b[0]
        self.values[:-self.layers[-1]] = ReLU(self.values[:-self.layers[-1]])
        acceleration = self.values[-2, 0]
        steering = self.values[-1, 0]
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
        self.W = [np.random.rand(self.out_size, self.total_size ) - 0.5]
        self.b = [np.random.rand(self.out_size, 1) - 0.5 ]
        return self
    
    def copy_weights(self, other):
        self.layers = other.layers
        self.W = [np.copy(w) for w in other.W]
        self.b = [np.copy(b) for b in other.b]
    
    def save(self, path):
        data = {'W':self.W, 'b':self.b}
        np.save("saved/mem_" + path, data)
        print('Model saved !')

    def load(self, path):
        data = np.load("saved/mem_" + path + ".npy", allow_pickle=True).item()
        self.W = data['W']
        self.b = data['b']
        print('Model loaded ! :', path)




def ReLU(x):
    return np.maximum(0, x)

def ReLU_der(x):
    return x > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))