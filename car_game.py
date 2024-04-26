

# a car game for AI to play

import pygame
import time
import numpy as np
import pymunk
import pymunk.pygame_util
import random
import time

from circuit import Circuit
from car import Car
from basic_ai import BasicAI, ManualController
from ai import AI_MODEL, AI

DEBUG = True


# Game class
class Game:
    def __init__(self, width=800, height=600, cars=[], circuit=None, hide_track=False, show_objects=True, screen=True, training=False):
        self.width = 800
        self.height = 600
        self.cars = cars

        self.objects = []
        self.game_time = 0
        self.reset_cars= False

        self.test = False
        self.show_objects = show_objects
        self.running = False
        self.SCREEN = screen
        self.show_screen = self.SCREEN
        self.training = training

        pygame.init()

        if self.SCREEN:
            self.clock = pygame.time.Clock()
            self.screen = pygame.display.set_mode((self.width, self.height))
            self.font = pygame.font.Font(None, 36)
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES

        # init pymunk
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.damping = 0.95

        # collision handler
        handler = self.space.add_collision_handler(1,2)

        # call wall_collide when the car collide with the wall
        handler.begin = Car.wall_collide
        handler.data['Game'] = self
        
        self.Circuit = Circuit(self.space, width=width, height=height, hide_circuit=hide_track)
        if circuit is None:
            # create the circuit
            self.Circuit.generate_circuit()
        else:
            self.Circuit.load(circuit)
        
            
        
        # add the cars to the space
        for car in self.cars:
            car.Game = self
            car.Circuit = self.Circuit
            car.activate()
            car.reset()
            


    def game_loop(self):
        pygame.display.set_caption('Cars')
        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                
                # restart the game
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.Circuit.remake()
                        for car in self.cars:
                            car.reset()
                            # if it is an ai
                            if isinstance(car.controller, AI):
                                car.controller.score = -1

                    if event.key == pygame.K_t:
                        self.test = not self.test

                    if event.key == pygame.K_a:
                        self.show_screen = not self.show_screen and self.SCREEN
                    # escape key
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

                    if event.key == pygame.K_i:
                        inp = float(input("new mutation_rate"))
                        for car in self.cars:
                            car.controller.mutation_rate = inp


            if self.show_screen:
                self.screen.fill((255, 255, 255))

            # get the actions
            for car in self.cars:
                car.get_input_state()
            for car in self.cars:
                car.update(test=self.test)
            
            if self.reset_cars:
                self.space.step(1/60)
                for car in self.cars:
                    car.reset()
                self.reset_cars = False
                
            
            # simulate the physics with small time steps
            steps = 2
            for i in range(steps):
                self.space.step(1/60/steps)

            self.game_time += 1

            if self.show_screen:
                # display objects
                if self.show_objects:
                    for i in range(len(self.objects)):
                        if self.game_time < self.objects[i][0]:
                            self.objects = self.objects[i:]
                            for i in range(len(self.objects)):
                                pygame.draw.circle(self.screen, (0, 0, 0, 0.3), self.objects[i][1], 1)
                            break
                    else: # if no break
                        self.objects = []
                
                # display with pymunk
                self.space.debug_draw(self.draw_options)

                # display fps
                text = self.font.render(str(int(self.clock.get_fps())), True, (0, 0, 0))
                self.screen.blit(text, (0, 0))

                pygame.display.update()
                if not self.test:
                    self.clock.tick(60) # 60 fps
                else:
                    self.clock.tick()
            else:
                pass
                #self.clock.tick() # no fps limit``

            if self.game_time % 120 == 0:
                print("#", end="")

        pygame.quit()



# main function
if __name__ == '__main__':

    manual_car = Car(ManualController(0), color=(255, 0, 0, 255))
    manual_car.show_distances = True
    manual_car2 = Car(ManualController(1), color=(255, 0, 100, 255))
    basic_AI = Car(BasicAI(), color=(0,0,255, 255))
    
    # load model "model.keras"
    
    ai = AI_MODEL(layers=[9, 4, 2])
    ai.load("model_1")
    ai_car = Car(ai, color=(0,255,0, 255))

    cars = [manual_car,manual_car2, ai_car, basic_AI]
    cars = [manual_car, ai_car]
    game = Game(cars=cars, circuit='circuit_1', hide_track=False) 
    game.game_loop()

