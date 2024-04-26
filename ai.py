
import numpy as np
import pygame
import time
import pymunk
import random

from car import Car
from basic_ai import Controller


class DeepReinforcementLearning:
    def __init__(self, best=None, train=False, layers=[8,4,3,2], n_duplicate=1):
        self.is_training = train
        self.n_duplicate = n_duplicate
        self.layers = layers
        self.best = best
        if best is not None:
            self.layers = best.layers
            self.W = [np.copy(w) for w in best.W]
            self.b = [np.copy(b) for b in best.b]
        self.Z = [None] * (len(self.layers)-1)
        self.A = [None] * (len(self.layers) - 1)

        self.n_layers = len(layers)
        self.score = 0
        self.score_added = 0

        #self.learning_rate = 1
        self.mutation_rate = 0.5
        self.mutation_chance = 0.4
        self.destroy = 0

    def forward_prop(self, input):
        self.Z[0] = self.W[0].dot(input) + self.b[0]
        for i in range(1, len(self.layers) - 1):
            self.A[i-1] = ReLU(self.Z[i-1])
            self.Z[i] = self.W[i].dot(self.A[i-1]) + self.b[i]
        self.A[-1] = (self.Z[-1])
        acceleration = self.A[-1][0, 0]
        steering = self.A[-1][1, 0]
        return acceleration, steering
        return self.Z[-1] # maybe should add sigmoid
    
    def mutate(self):
        for j in range(len(self.W)):
            self.W[j] = self.best.W[j] + self.get_random_variation(self.W[j].shape[0], self.W[j].shape[1])
            self.b[j] = self.best.b[j] + self.get_random_variation(self.b[j].shape[0], self.b[j].shape[1]) * 0.2
            
            self.W[j] = np.clip(self.W[j], -1, 1)
            self.b[j] = np.clip(self.b[j], -1, 1)

    def get_random_variation(self, x, y):
        mask = np.random.rand(x, y) < self.mutation_chance # only mutate 10% of the weights
        return self.mutation_rate * (np.random.rand(x,y) * 2 - 1) * mask
        
    def update_score(self, score):
        self.score_added += 1
        self.score = (self.score * (self.score_added - 1) + score) / self.score_added
        if self.score_added == self.n_duplicate and self.best is not None:
            # update the model
            self.best.update_model(self)

    def init(self):
        self.W = [np.random.rand(self.layers[i+1], self.layers[i]) - 0.5 for i in range(len(self.layers)-1)]
        self.b = [np.random.rand(self.layers[i+1], 1) - 0.5 for i in range(len(self.layers)-1)]
        return self


class AI (Controller, DeepReinforcementLearning):
    def __init__(self, train=False, id=-1, best=None, layers=[8,4,3,2], n_duplicate=1):
        # call the constructors of the parent classes
        Controller.__init__(self)
        DeepReinforcementLearning.__init__(self, best, train, layers, n_duplicate)

        self.cars = []
        self.car = -1 # this says to use multiple cars
        self.id = id

    def update(self):
        pass

    def get_input(state):
        distances = [state['distances'][i] / 400 for i in range(len(state['distances']))]
        directions = [state['directions'][i] for i in range(len(state['directions']))]
        # , state['state'] / 100 
        input = np.array([distances + directions + [state['speed'] / 300]]).T
        input = np.clip(input, -1, 1)
        return input

    def get_action(self, state):
        self.update()
        input = AI.get_input(state) # get the input

        acceleration, steering = self.forward_prop(input) # get the actions
        return acceleration, steering
    
    def wall_collide(self, car):
        self.compute_score(car)
        if self.best is not None:
            car.deactivate()
    
    def lap(self, car):
        self.update_score(1000 - car.get_lap_game_time() / 10)
        if self.best is None:
            print('Lap finished : ', car.state, car.get_lap_game_time() / 60, "s")
        else:
            car.deactivate()
            #self.best.new_generation()
    
    def compute_score(self, car):
        score = car.state + 1 - np.linalg.norm(car.body.position - car.Circuit.circuit[car.state]) /  np.linalg.norm(car.Circuit.circuit[(car.state+1) % car.Circuit.n_points] - car.Circuit.circuit[car.state])
        score -= car.get_lap_game_time() / 500
        self.update_score(score)



class AI_MODEL(AI): # for the best car
    def __init__(self, train=False, id=-1, best=None, layers=[4,3], n_variations=100, n_duplicates=1):
        super().__init__(train, id, best, layers)
        self.n_variations = n_variations
        self.n_duplicates = n_duplicates
        self.variations = []
        self.var_cars = []
        self.total_runs = 0
        self.model_updates = 0
        self.generation = 0
        self.generation_time = time.time()
        self.generation_game_time = 0
        self.car = None
        
        self.BY_GENERATION = train
        self.REMAKE_CIRCUIT = False

    def create_variations(self):
        self.variations = [self]
        if self.is_training:
            for i in range(1, self.n_variations):
                variation = AI(train=self.is_training, id=i, best=self, layers=self.layers, n_duplicate=self.n_duplicates)
                variation.mutate()
                self.variations.append(variation)
    

    def update_model(self, other): # as the best car
        self.total_runs += 1
        if self.BY_GENERATION:
            # deactivate the car body to avoid collision
            for car in other.cars:
                car.deactivate()
            self.model_updates += 1
            #print(f"Model updates : {self.model_updates}/{ len(self.variations)-1}  by car : {other.id}")
        else:
            if other.score >= self.score:
                self.W = [np.copy(w) for w in other.W]
                self.b = [np.copy(b) for b in other.b]
                self.score = other.score
                self.cars[0].reset()
                print(f"{self.total_runs} new best score :{self.score}")
    
    def update(self):
        if self.BY_GENERATION:
            self.check_generation()
        # show the weights of the model
        if self.is_training and self.best is None and self.car.Game.show_screen:
            self.show()
        return super().update()

    def check_generation(self):   # check if the generation is finished
        #print(f"Model updates : {self.model_updates}/{ len(self.variations)-1}  {self.total_runs}")
        if self.model_updates >= len(self.variations) - 1 or (self.car.Game.game_time - self.generation_game_time) > 60 * 25:
            if (self.car.Game.game_time - self.generation_game_time) > 60 * 25:
                print("Time out")
            self.new_generation()
    
    def new_generation(self):
        # create a new generation
        self.model_updates -= len(self.variations) - 1
        self.generation += 1

        # calculate the score of the cars
        for i in range(len(self.var_cars)):
            if self.var_cars[i].is_active:
                self.compute_score(self.var_cars[i])
        
        score = -1
        best = 0
        # get the best car and update the model
        for i in range(len(self.variations)):
            if self.variations[i].score > score:
                score = self.variations[i].score
                best = self.variations[i]
            self.variations[i].score = 0
            self.variations[i].score_added = 0
        
        # update the model
        if score > 2:
            self.W = [np.copy(w) for w in best.W]
            self.b = [np.copy(b) for b in best.b]


        if self.REMAKE_CIRCUIT:
            self.car.Circuit.remake()

        t = time.time() - self.generation_time
        gt = (self.car.Game.game_time - self.generation_game_time) / 60
        speed = gt / t
        self.generation_time = time.time()
        self.generation_game_time = self.car.Game.game_time
        print(f"Generation {self.generation} best score : {score} time : {t} game time : {gt} speed : {speed}")

        for ai in self.variations[1:]:
            ai.mutate()
            # activate car body to be able to move
            for car in ai.cars:
                #print(car, car.is_active)
                #car.reset()
                car.activate()
        self.car.Game.reset_cars = True
        

    def save(self, path):
        # this is a very simple way to save the model
        data = {'W':self.W, 'b':self.b}
        np.save(path, data)
        print('Model saved !')

    def load(self, path):
        data = np.load(path + ".npy", allow_pickle=True).item()
        self.W = data['W']
        self.b = data['b']
        print('Model loaded ! :', path)

    def show(self):
        # show the neurons and weigts on the screen
        screen = self.car.Game.screen
        
        # make circles for the neurons and lines for the weights
        alpha = 100
        position = lambda x, y: (int(150 + x * 100), int(250 + y *200 / (self.W[0].shape[1] if x==0 else self.W[x-1].shape[0])))
        for i in range(len(self.W)):
            for j in range(self.W[i].shape[1]):
                for k in range(self.W[i].shape[0]):
                    size = int(abs(self.W[i][k, j]*4))
                    if self.W[i][k, j] > 0:
                        pygame.draw.line(screen, (255, 0, 0, alpha), position(i, j), position(i+1, k), size)
                    else:
                        pygame.draw.line(screen, (0, 0, 255, alpha), position(i, j), position(i+1, k), size)
                pygame.draw.circle(screen, (0, 0, 0), position(i, j), 6)
                pygame.draw.circle(screen, (255, 255, 255), position(i, j), 3)
            for j in range(self.W[i].shape[0]):
                pygame.draw.circle(screen, (0, 0, 0), position(i+1, j), 6)
                pygame.draw.circle(screen, (255, 255, 255), position(i+1, j), 3)
                
    def get_cars(self):
        self.var_cars = [Car(self.variations[i], 
                             color=(0, 100 + 130 * i / len(self.variations), i*47%256,150), 
                             offset=[0, (j- self.n_duplicates/2)*4]) 
                         for j in range(self.n_duplicates) for i in range(1, len(self.variations))]
        return [Car(self.variations[0], color=(0,255,0,255))] + self.var_cars

def ReLU(x):
    return np.maximum(0, x)

def ReLU_der(x):
    return x > 0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))








