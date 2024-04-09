
import numpy as np
import pygame
import time
import pymunk
import pymunk.pygame_util

DEBUG = False

# car class
class Car:
    MAX_STATIC_FRICTION = 150000
    MAX_STATIC_FRICTION_SQ = MAX_STATIC_FRICTION ** 2
    DYNAMIC_FRICTION = 70000

    def __init__(self, controller, mass=1000, color=(0, 0, 255,255)):
        #self.pos = np.zeros(2)
        #self.velocity = np.zeros(2)
        #self.rotation = 0
        #self.rotation_velocity = 0

        self.speed = 0


        self.MAX_STEERING = np.pi / 6
        

        self.WIDTH = 20
        self.HEIGHT = 8
        controller.car = self
        self.controller = controller
        self.color = color
        self.show_distances = False


        self.is_active = True
        
        
        self.SIDE_FRICTION = 10000

        self.MOTOR_FORCE = lambda x : 30000 * x if x * self.speed > 0 else 100000 * x

        #self.image = pygame.image.load('car.png')
        #self.image.fill((0, 0, 0, 255), special_flags=pygame.BLEND_ADD) # add the color to the image
        self.Game = None
        self.Circuit = None

        # pymunk
        self.MOMENT_INERTIE = 100000
        self.body = pymunk.Body(mass, self.MOMENT_INERTIE)
        #self.body.position = (0, 0)
        #self.body.center_of_gravity = (0, 0)
        self.shape = pymunk.Poly(self.body, [(-self.WIDTH / 2, -self.HEIGHT / 2), (self.WIDTH / 2, -self.HEIGHT / 2), (self.WIDTH / 2, self.HEIGHT / 2), (-self.WIDTH / 2, self.HEIGHT / 2)])
        self.shape.color = color
        self.shape.friction = 0.5
        self.shape.elasticity = 0.2
        self.shape.collision_type = 1


        # filter : only collide with the circuit
        self.shape.filter = pymunk.ShapeFilter(categories=0b1, mask=0b10)

    def FRICTION(motor, side): 
        sq = motor**2 + side**2
        if sq < Car.MAX_STATIC_FRICTION_SQ: 
            return (motor, side, False) 
        else:
            coeff = Car.DYNAMIC_FRICTION / np.sqrt(sq)
            return (motor * coeff, side * coeff, True)
        
    def get_lap_game_time(self):
        return self.Game.game_time - self.start_game_time

    def update(self, dt=1/60, test = False):
        if not self.is_active:
            return
        acceleration, steering = self.get_action()

        ### CAR PHYISICS ###
        front_wheel_offset = np.array([self.WIDTH / 2, 0])
        back_wheel_offset = np.array([-self.WIDTH / 2, 0])
        rotation = np.array([[np.cos(self.body.angle), -np.sin(self.body.angle)], [np.sin(self.body.angle), np.cos(self.body.angle)]])

        # motor force
        def get_motor_and_side_friction(angle, offset):
            rotated_offset = np.dot(rotation, offset)

            wheel_direction = np.array([np.cos(angle), np.sin(angle)]) # direction of the wheel
            motor_force_magnitude = self.MOTOR_FORCE(acceleration)           # magnitude of the motor
            

            side_angle = angle + np.pi / 2
            wheel_velocity = self.body.velocity + np.cross([0, 0, self.body.angular_velocity], [rotated_offset[0], rotated_offset[1], 0])[:2] # velocity of the wheel
            wheel_side_direction = np.array([np.cos(side_angle), np.sin(side_angle)])    # side direction
            wheel_side_friction_magnitude = np.dot(wheel_velocity, wheel_side_direction)  * self.SIDE_FRICTION # side velocity * side friction



            motor_force_magnitude, wheel_side_friction_magnitude, derapage = Car.FRICTION(motor_force_magnitude, wheel_side_friction_magnitude)



            motor_force = wheel_direction * motor_force_magnitude
            wheel_side_friction = wheel_side_direction * ( - wheel_side_friction_magnitude)
            if derapage and self.Game.show_screen and self.Game.show_objects:
                t = self.Game.game_time + 60
                self.Game.objects.append((t, self.body.position + np.dot(rotation, offset + np.dot(rotation,np.array([0, self.HEIGHT / 2])))))
                self.Game.objects.append((t, self.body.position + np.dot(rotation, offset + np.dot(rotation,np.array([0, - self.HEIGHT / 2])))))

            #self.body.apply_force_at_world_point(list(motor + wheel_side_friction), list(rotated_offset + self.body.position))
            self.body.apply_impulse_at_world_point(list((motor_force + wheel_side_friction)/60) , list(rotated_offset + self.body.position))
            
            return motor_force, wheel_side_friction
        
        front_wheel_angle = self.body.angle + steering * self.MAX_STEERING
        back_wheel_angle = self.body.angle
        front_motor, front_wheel_side_friction = get_motor_and_side_friction(front_wheel_angle, front_wheel_offset)
        back_motor, back_wheel_side_friction = get_motor_and_side_friction(back_wheel_angle, back_wheel_offset)

        # apply the forces
        if not test and DEBUG:
            # show the forces in red
            pygame.draw.line(self.Game.screen, (200, 0, 0), self.body.position + front_wheel_offset, self.body.position + front_wheel_offset + front_motor * 0.001)
            pygame.draw.line(self.Game.screen, (200, 0, 0), self.body.position + back_wheel_offset , self.body.position + back_wheel_offset  + back_motor * 0.001)

            pygame.draw.line(self.Game.screen, (0, 100, 0), self.body.position + front_wheel_offset, self.body.position + front_wheel_offset + front_wheel_side_friction * 0.0001)
            pygame.draw.line(self.Game.screen, (0, 100, 0), self.body.position + back_wheel_offset , self.body.position + back_wheel_offset  + back_wheel_side_friction * 0.0001)
        

        self.speed = np.linalg.norm(self.body.velocity) * np.sign(np.dot(self.body.velocity, np.array([np.cos(self.body.angle), np.sin(self.body.angle)])))

        ### check points ### 
        # if the car is on front of the next checkpoint
        # dot product : if the dot product is positive, the car is in front of the checkpoint


        vect = self.Circuit.circuit[self.state + 1] - self.Circuit.circuit[self.state]
        while np.dot(self.body.position - self.Circuit.circuit[self.state], vect) > 0:
            self.state = (self.state + 1)
            if self.state + 1 >= len(self.Circuit.circuit):
                self.controller.lap()
                self.reset()
            vect = self.Circuit.circuit[self.state + 1] - self.Circuit.circuit[self.state]


    
    def reset(self):
        # reset the car
        # reset the position
        self.body.velocity = (0, 0)
        self.body.angular_velocity = 0
        self.body.position = list(self.Circuit.circuit[0])
        # nice orientation
        self.body.angle = np.arctan2(self.Circuit.circuit[1][1] - self.Circuit.circuit[0][1], self.Circuit.circuit[1][0] - self.Circuit.circuit[0][0])
        
        self.speed = 0
        self.state = 0
        self.start_game_time = self.Game.game_time
        self.start_time = time.time()
        self.controller.destroy = 0


    def get_input_state(self):
        if not self.is_active:
            return 0, 0
        self.input_state = {'state':self.state, 'speed':self.speed, 'rotation':self.body.angle}

        # get the distance with the walls
        angles = np.array([-40, -15, 0, 15, 40]) * np.pi / 180

        distances = [400] * len(angles)

        for i in range(len(angles)):
            # cast a ray with pymunk
            direction = np.array([np.cos(self.body.angle + angles[i]), np.sin(self.body.angle + angles[i])])
            shape_filter = pymunk.ShapeFilter(categories=0b1, mask=0b10)
            info = self.Game.space.segment_query_first(list(self.body.position), list(self.body.position + direction * distances[i]), 2, shape_filter)
            if info is not None:
                distances[i] = np.linalg.norm(info.point - self.body.position)
            
            if self.show_distances:
                #pygame.draw.line(self.Game.screen, (100, 0, 0), self.body.position, self.body.position + direction * distances[i], 1)
                pygame.draw.circle(self.Game.screen, (0, 0, 0), self.body.position + direction * distances[i], 4)

        self.input_state['distances'] = distances
        return self.input_state
    
    def get_action(self):
        actions = self.controller.get_action(state=self.input_state)
        actions = np.clip(actions, -1, 1)
        return actions

    # print fuction
    def __str__(self):
        return f'Car : pos={self.body.position}, angle={self.body.angle}, state={self.state}'
    
    def wall_collide(arbiter, space, data):
        
        
        body = arbiter.shapes[0].body
        car = None
        for car in data['Game'].cars:
            if car.body == body:
                break
        if not car.is_active:
            return False
        
        car.controller.wall_collide()
        car.reset()
        return True
    
