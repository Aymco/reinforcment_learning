


import pygame
import numpy as np

# base interface for controller
class Controller:
    def __init__(self):
        self.car = None

    def get_action(self, state):
        pass

    def wall_collide(self):
        pass

    def lap(self):
        print('Lap finished : ', self.car.state, self.car.get_lap_game_time() / 60, "s")




# manual controller class
class ManualController (Controller):
    KEYS = [[pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT],
            [pygame.K_z, pygame.K_s, pygame.K_q, pygame.K_d]]
    def __init__(self, player=0):
        self.player = player
        self.car = None
        self.steering = 0

    def get_action(self, state):
        #if self.player == 0:
        #    print(state['distances'])
        keys = pygame.key.get_pressed()
        acceleration = 0
        steering = 0
        if keys[self.KEYS[self.player][0]]:
            acceleration = 1
        if keys[self.KEYS[self.player][1]]:
            acceleration = -1
        if keys[self.KEYS[self.player][2]]:
            steering = -1
        if keys[self.KEYS[self.player][3]]:
            steering = 1
        # smooth the steering with a lerp function
        self.steering = 0.95 * self.steering + 0.05 * steering
        return (acceleration, self.steering)



# basic ai for car_game
class BasicAI (Controller):
    def __init__(self):
        self.car = None

    def get_action(self, state):
        distances = state['distances']
        acceleration = (distances[2] - 40)/400
        steering = (distances[3] - distances[1]) / 50
        
        return acceleration, steering

    def wall_collide(self):
        pass