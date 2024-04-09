
import numpy as np
import pygame
import time
import pymunk
import random

from car import Car
from basic_ai import Controller

## reinforcment learning ai
class AI (Controller):
    number_of_cars = 1

    def __init__(self, train=False, id=-1, best=None, layers=[8,4,3,2]):
        self.car = None
        self.is_training = train
        self.id = id
        self.best = best

        self.layers = layers
        if best is not None:
            self.layers = best.layers
            self.W = [np.copy(w) for w in best.W]
            self.b = [np.copy(b) for b in best.b]


        self.n_layers = len(layers)
        self.score = -1

        #self.learning_rate = 1
        self.mutation_rate = 1
        self.destroy = 0

        self.Z = [None] * (len(self.layers)-1)
        self.A = [None] * (len(self.layers) - 1)
    
    def update(self):
        if self.is_training:
            if abs(self.car.speed) < 10:
                self.destroy += 1
                if self.destroy > 30:
                    self.end_of_run()
                    self.car.reset()
                    return 0, 0
            else:
                self.destroy = 0


    def get_input(state):
        distances = [state['distances'][i] / 400 for i in range(len(state['distances']))]
        if state['speed'] > 300:
            print('speed :', state['speed'])
        input = np.array([distances + [state['speed'] / 300, state['state'] / 100 ]]).T
        input = np.clip(input, -1, 1)
        return input

    def get_action(self, state):
        self.update()

        input = AI.get_input(state) # get the input

        acceleration, steering = self.forward_prop(input) # get the actions
        return acceleration, steering
    
    def wall_collide(self):
        self.end_of_run()
    
    def lap(self):
        if self.best is None:
            print('Lap finished : ', self.car.state, self.car.get_lap_game_time() / 60, "s")
        self.end_of_run(lap=True)
    
    def end_of_run(self, lap=False):
        if lap:
            score = 1000 - self.car.get_lap_game_time() / 10
        else:
            score = self.car.state + 1 - np.linalg.norm(self.car.body.position - self.car.Circuit.circuit[self.car.state]) /  np.linalg.norm(self.car.Circuit.circuit[(self.car.state+1) % self.car.Circuit.n_points] - self.car.Circuit.circuit[self.car.state])
            score -= self.car.get_lap_game_time() / 500


        self.score = score
        if self.best is not None:
            # update the model
            self.best.update_model(self)
    
    def forward_prop(self, input):
        self.Z[0] = self.W[0].dot(input) + self.b[0]
        #self.A[0] = ReLU(self.Z[0])
        for i in range(1, len(self.layers) - 1):
            self.A[i-1] = ReLU(self.Z[i-1])
            self.Z[i] = self.W[i].dot(self.A[i-1]) + self.b[i]

        self.A[-1] = self.Z[-1] 
        # TODO see last function ??
        acceleration = self.A[-1][0, 0]
        steering = self.A[-1][1, 0]
        
        return acceleration, steering
    
    
    
    def mutate(self):
        for j in range(len(self.W)):
            self.W[j] = self.best.W[j] + self.get_random_variation(self.W[j].shape[0], self.W[j].shape[1])
            self.b[j] = self.best.b[j] + self.get_random_variation(self.b[j].shape[0], self.b[j].shape[1]) * 0.2
            
            self.W[j] = np.clip(self.W[j], -1, 1)
            self.b[j] = np.clip(self.b[j], -1, 1)

    def get_random_variation(self, x, y):
        mask = np.random.rand(x, y) < 0.1 # only mutate 10% of the weights
        return self.mutation_rate * ((np.random.rand(x,y)*2-1) * mask)


class AI_MODEL(AI): # for the best car
    def __init__(self, train=False, id=-1, best=None, layers=[4,3], n_variations=100):
        super().__init__(train, id, best, layers)
        self.n_variations = n_variations
        self.variations = []
        self.total_runs = 0
        self.model_updates = 0
        self.generation = 0
        self.generation_time = time.time()
        self.generation_game_time = 0
        
        self.BY_GENERATION = train
        self.REMAKE_CIRCUIT = False
        
    def init(self):
        self.W = [np.random.rand(self.layers[i+1], self.layers[i]) - 0.5 for i in range(len(self.layers)-1)]
        self.b = [np.random.rand(self.layers[i+1], 1) - 0.5 for i in range(len(self.layers)-1)]
        return self


    def create_variations(self):
        self.variations = [self]
        if self.is_training:
            for i in range(1, self.n_variations):
                variation = AI(train=self.is_training, id=i, best=self)
                variation.mutate()
                self.variations.append(variation)
    

    def update_model(self, other): # as the best car
        self.total_runs += 1
        if self.BY_GENERATION:
            # deactivate the car body to avoid collision
            self.car.Game.space.remove(other.car.body)
            self.car.Game.space.remove(other.car.shape)
            other.car.is_active = False
            self.model_updates += 1   
        else:
            if other.score >= self.score:
                self.W = [np.copy(w) for w in other.W]
                self.b = [np.copy(b) for b in other.b]
                self.score = other.score
                self.car.reset()
            
                print(f"{self.total_runs} new best score :{self.score}")
    
    def update(self):
        if self.BY_GENERATION:
            self.check_generation()
        # show the weights of the model
        if self.is_training and self.best is None and self.car.Game.show_screen:
            self.show()
        return super().update()

    def check_generation(self):
        #print(f"Model updates : {self.model_updates}/{ len(self.variations)-1}  {other.id}  {self.total_runs}")
        if self.model_updates >= len(self.variations)-1 or (self.car.Game.game_time - self.generation_game_time) > 60 * 25:
            if (self.car.Game.game_time - self.generation_game_time) > 60 * 25:
                print("Time out")
            # new generation
            self.model_updates -= len(self.variations)-1
            self.generation += 1
            if self.REMAKE_CIRCUIT:
                self.car.Circuit.remake()

            # get the best car and update the model
            score = -1
            best = 0
            for i in range(len(self.variations)):
                if self.variations[i].car.is_active:
                    self.variations[i].end_of_run()
                if self.variations[i].score > score:
                    score = self.variations[i].score
                    best = self.variations[i]
            if score > 5:
                self.W = [np.copy(w) for w in best.W]
                self.b = [np.copy(b) for b in best.b]

            t = time.time() - self.generation_time
            gt = (self.car.Game.game_time - self.generation_game_time) / 60
            speed = gt / t
            self.generation_time = time.time()
            self.generation_game_time = self.car.Game.game_time
            print(f"Generation {self.generation} best score : {score} time : {t} game time : {gt} speed : {speed}")

            for ai in self.variations[1:]:
                ai.mutate()
                # activate car body to be able to move
                if not ai.car.is_active:
                    self.car.Game.space.add(ai.car.body, ai.car.shape)
                    ai.car.is_active = True
                ai.car.reset()
            self.car.reset()

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
        cars = [Car(self.variations[i], color=(0,150,0,150)) for i in range(1, len(self.variations))]
        return [Car(self.variations[0], color=(0,255,0,255))] + cars
         

def ReLU(x):
    return np.maximum(0, x)

def ReLU_der(x):
    return x > 0

def Softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)








