
import numpy as np
import pymunk
import random
from scipy import interpolate

class Circuit:
    def __init__(self, space, width=800, height=600, road_width=50, n_points=100, hide_circuit=False):
        self.width = width
        self.height = height

        self.road_width = road_width
        self.n_points = n_points
        self.space = space
        self.hide_circuit = hide_circuit


    def add_to_space(self):
        # add the circuit to the space
        # left and right borders of the circuit
        for i in range(len(self.circuit) - 1):
            segment = pymunk.Segment(self.space.static_body, list(self.left[i]), list(self.left[i + 1]), 10)
            segment.collision_type = 2
            segment.elasticity = 0
            segment.friction = 1
            segment.color = (0, 0, 0, 255)
            if self.hide_circuit:
                segment.color = (255, 255, 255, 0)
            # static
            segment.body.body_type = pymunk.Body.STATIC
            self.space.add(segment)
            self.left[i] = segment

            segment = pymunk.Segment(self.space.static_body, list(self.right[i]), list(self.right[i + 1]), 10)
            segment.collision_type = 2
            segment.elasticity = 0
            segment.friction = 1
            segment.color = (0, 0, 0, 255)
            if self.hide_circuit:
                segment.color = (255, 255, 255, 0)
            self.space.add(segment)
            self.right[i] = segment

    def generate_circuit(self):
        self.circuit = []
        #base_points_count = 6
        #base_points = [(random.randint(self.road_width, self.width-self.road_width), random.randint(self.road_width, self.height-self.road_width)) for i in range(base_points_count)]
        
        base_points = [(1,2),(3,2),(5,2),(7,2),(7,6),(5,6),(3,6),(1,6)]
        base_points = [(x + random.random()*1 - 0.5, y  + random.random()*3 - 1.5) for x, y in base_points]     
        base_points = [((x+0.5) * self.width /9, (y+0.5) * self.height / 9) for x, y in base_points]
        
        # sort the points to have a non crossing circuit
        base_points = base_points + base_points[:3] # add the first 3 points to the end

        # interpolate the points 
        index = [i for i in range(len(base_points))]
        x = [point[0] for point in base_points]
        y = [point[1] for point in base_points]

        # quadratic approximation (no extrapolation)
        cs_x = interpolate.interp1d(index, x, kind='quadratic')
        cs_y = interpolate.interp1d(index, y, kind='quadratic')


        # generate the circuit
        index_new = np.linspace(1, len(base_points) - 2, self.n_points)
        x_new = cs_x(index_new)
        y_new = cs_y(index_new)
        self.circuit = np.array([(x_new[i], y_new[i]) for i in range(self.n_points)])

        self.generate_borders()
        self.add_to_space()

    def generate_borders(self):
        self.left = [None] * len(self.circuit)
        self.right = [None] * len(self.circuit)
        for i in range(len(self.circuit) - 1):
            vect = self.circuit[i+1] - self.circuit[i]
            vect = vect / np.linalg.norm(vect)
            vect = np.array([-vect[1], vect[0]])
            self.left[i] = self.circuit[i] + vect * self.road_width
            self.right[i] = self.circuit[i] - vect * self.road_width

            # check if left is on the left of the last left
            if i != 0:
                if np.cross(self.left[i] - self.circuit[i], self.left[i - 1] - self.circuit[i]) < 0:
                    self.left[i] = self.left[i - 1]

                if np.cross(self.right[i] - self.circuit[i], self.right[i - 1] - self.circuit[i]) > 0:
                    self.right[i] = self.right[i - 1]

                if np.cross(self.right[i] - self.left[i-1], self.left[i] - self.left[i-1]) < 0:
                    self.left[i] = self.left[i - 1]

                if np.cross(self.left[i] - self.right[i-1], self.right[i] - self.right[i-1]) > 0:
                    self.right[i] = self.right[i - 1]

        self.left[-1] = self.left[0]
        self.right[-1] = self.right[0]

    def remake(self):
        # clear the old circuit from the space
        for i in range(len(self.circuit) - 1):
            self.space.remove(self.left[i])
            self.space.remove(self.right[i])
        
        # generate a new circuit
        self.generate_circuit()
    

    def save(self, path):
        # save the circuit with numpy
        # save n_points, width, height, road_width, circuit
        data = {'n_points':self.n_points, 'width':self.width, 'height':self.height, 'road_width':self.road_width, 'circuit':self.circuit}
        np.save(path, data)
        # doesn't work with because inhomogeneous list

    
    def load(self, path):
        # load the circuit
        data = np.load(path + ".npy", allow_pickle=True).item()
        self.n_points = data['n_points']
        self.width = data['width']
        self.height = data['height']
        self.road_width = data['road_width']
        self.circuit = data['circuit']

        self.generate_borders()
        self.add_to_space()
        return self