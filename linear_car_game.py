

# a car game for AI to play

import time
import numpy as np
import random
import time

from linear_car import Car
#from basic_ai import BasicAI, ManualController
#from ai import AI_MODEL, AI

DEBUG = True

# base interface for controller
class Controller:
    def __init__(self):
        self.car = None

    def get_action(self, state):
        pass

    def lap(self, car):
        print('Lap finished : ', self.car.state, self.car.get_lap_game_time() / 60, "s")



class DumbController (Controller):
    def get_action(self, state):
        return (1, 0)
    


# Game class
class Game:
    def __init__(self, width=800, circuit_len=600, cars=[]):
        self.width = 50
        self.circuit_len = 500
        self.road_width = 5

        self.cars = cars

        self.training = False

        self.objects = []
        self.game_time = 0
        self.reset_cars= False

        self.test = False
        self.running = False

        self.generate_circuit()
        self.draw_circuit_terminal() 
        
        # add the cars to the space
        for car in self.cars:
            car.Game = self
            car.start_position = self.circuit[0], self.width / 2
            car.activate()
            car.reset()

        
        self.game_loop()    

    def generate_circuit(self):
        # circuit is a list of y values
        self.circuit = np.zeros(self.circuit_len) +  self.width / 2

        self.number_of_sinus = 4

        # random amplitude, frequency and phase
        self.amplitudes = (np.random.rand(self.number_of_sinus) + 1) / 2 * (self.width - self.road_width) / 2 / self.number_of_sinus
        self.frequencies = np.random.rand(self.number_of_sinus) * 0.3 + 0.05
        self.phases = np.random.rand(self.number_of_sinus) * 2 * np.pi

        for i in range(self.circuit_len):
            sin_value = self.amplitudes * np.sin(self.frequencies * i + self.phases)
            self.circuit[i] += np.sum(sin_value)

        self.c_delta = np.diff(self.circuit)# the derivative of the circuit
        self.c_delta = np.concatenate(([self.c_delta[0]] , self.c_delta))

        # smooth the derivative
        smooth_size = 10
        add_to_start = smooth_size // 2
        self.c_delta = np.concatenate(([self.c_delta[0]] * (add_to_start), self.c_delta, [self.c_delta[-1]] * (smooth_size - add_to_start - 1)))
        self.c_delta = np.convolve(self.c_delta, np.ones(smooth_size) / smooth_size, mode='valid')


        self.c_right = self.circuit + self.road_width * np.sqrt(1 + self.c_delta ** 2)
        self.c_left = 2 * self.circuit - self.c_right


    def draw_circuit_terminal(self, terminal_size=50):
        rapport = self.width / terminal_size

        window_size = 10
        start = int(self.cars[0].position[1] - window_size // 2)


        for i in range(start, start + window_size):
            y = int(i * rapport)
            print("|", end="")
            for j in range(terminal_size):
                x = j * rapport
                for car in self.cars:
                    if abs(x - car.position[0]) + abs(y - car.position[1]) < 1:
                        print("X", end="")
                        break
                else:

                    if abs(x - self.circuit[i]) < 0.5:
                        print(".", end="")
                    if self.c_left[y] < x < self.c_right[y]:
                        print("#", end="")
                    else:
                        print(" ", end="")
            print("|", y)
        print("")
        for car in self.cars:
            print(car)

    def check_collisions(self):
        for car in self.cars:
            if car.destroy_countdown > 0:
                car.destroy_countdown -= 1
            else:
                for i in range(0, len(car.points), 2):
                    x = car.points[i]
                    y = car.points[i + 1]
                    if self.c_left[int(x)] > y or self.c_right[int(x)] < y:
                        car.destroy_countdown = 60
                        car.deactivate()
                        break

    def update_cars(self):
        # get the actions
        for car in self.cars:
            car.get_input_state()

        for car in self.cars:
            car.update(test=self.test)

        
    def game_loop(self):
        self.running = True
        while self.running:
            

            self.update_cars()
            
            if self.reset_cars:
                for car in self.cars:
                    car.reset()
                self.reset_cars = False
                
            
            self.game_time += 1
            if self.game_time % 120 == 0:
                self.draw_circuit_terminal() 


# main function
if __name__ == '__main__':

    manual_car = Car(DumbController(), color=(255, 0, 0, 255))
    #manual_car2 = Car(ManualController(1), color=(255, 0, 100, 255))
    #basic_AI = Car(BasicAI(), color=(0,0,255, 255))
    
    # load model "model.keras"
    
    cars = [manual_car]
    game = Game(cars=cars)



