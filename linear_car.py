
import numpy as np

import time

DEBUG = False

# car class
class Car:
    MAX_STATIC_FRICTION = 250000
    MAX_STATIC_FRICTION_SQ = MAX_STATIC_FRICTION ** 2
    DYNAMIC_FRICTION = 25000
    WIDTH = 20
    HEIGHT = 8
    MAX_STEERING = np.pi / 6
    SIDE_FRICTION = 10000

    def __init__(self, controller, MASS=1000, color=(0, 0, 255,255), start_position=(0, 0)):
        self.is_active = False
        self.start_position = start_position

        self.MASS = MASS
        self.MOMENT_INERTIE = 100000
        self.MOTOR_FORCE = lambda x : 50000 * x if x * self.speed > 0 else 200000 * x

        
        # if controller is AI
        if controller.car == -1:
            controller.cars.append(self)
        else:
            controller.car = self

        self.controller = controller
        self.color = color

        self.Game = None

        self.reset()


    def FRICTION(motor, side): 
        sq = motor**2 + side**2
        if sq < Car.MAX_STATIC_FRICTION_SQ: 
            return (motor, side, False) 
        else:
            coeff = Car.DYNAMIC_FRICTION / np.sqrt(sq)
            return (motor * coeff, side * coeff, True)
        
    def get_lap_game_time(self):
        return self.Game.game_time - self.start_game_time

    def update(self, dt=1/60, test=False):
        if not self.is_active:
            return
        acceleration, steering = self.get_action()


        if self.Game.training:
            if abs(self.speed) < 10:
                self.destroy_countdown += 1
                if self.destroy_countdown > 30:
                    self.controller.wall_collide(self)
                    self.reset()
                    return
            else:
                self.destroy_countdown = 0

        self.car_physics(dt, acceleration, steering)

        # maybe update the state
        ### check points ### 


    def car_physics(self, dt, acceleration, steering):
        ### CAR PHYISICS ###
        front_wheel_offset = np.array([self.WIDTH / 2, 0])
        back_wheel_offset = np.array([-self.WIDTH / 2, 0])
        rotation = np.array([[np.cos(self.angle), -np.sin(self.angle)], [np.sin(self.angle), np.cos(self.angle)]])

        # motor force
        def get_motor_and_side_friction(angle, offset, motor_force=1):
            rotated_offset = np.dot(rotation, offset)

            wheel_direction = np.array([np.cos(angle), np.sin(angle)]) # direction of the wheel
            motor_force_magnitude = self.MOTOR_FORCE(acceleration) * motor_force        # magnitude of the motor
            
            side_angle = angle + np.pi / 2
            wheel_velocity = self.velocity + np.cross([0, 0, self.angular_velocity], [rotated_offset[0], rotated_offset[1], 0])[:2] # velocity of the wheel
            wheel_side_direction = np.array([np.cos(side_angle), np.sin(side_angle)])    # side direction
            wheel_side_friction_magnitude = np.dot(wheel_velocity, wheel_side_direction) * self.SIDE_FRICTION # side velocity * side friction

            motor_force_magnitude, wheel_side_friction_magnitude, derapage = Car.FRICTION(motor_force_magnitude, wheel_side_friction_magnitude)

            motor_force = wheel_direction * motor_force_magnitude
            wheel_side_friction = wheel_side_direction * ( - wheel_side_friction_magnitude)

            # apply the forces
            self.velocity += (motor_force + wheel_side_friction) / self.MASS * dt
            self.angular_velocity += np.cross(rotated_offset, motor_force) / self.MOMENT_INERTIE * dt
            
            return motor_force, wheel_side_friction
        
        front_wheel_angle = self.angle + steering * self.MAX_STEERING
        back_wheel_angle = self.angle
        get_motor_and_side_friction(front_wheel_angle, front_wheel_offset, motor_force=0)
        get_motor_and_side_friction(back_wheel_angle, back_wheel_offset)

        # apply the forces
        self.speed = np.linalg.norm(self.velocity) * np.sign(np.dot(self.velocity, np.array([np.cos(self.angle), np.sin(self.angle)])))

    
    def reset(self):
        # reset the car
        self.position = self.start_position
        self.angle = 0
        self.velocity = (0, 0)
        self.angular_velocity = 0
        #self.force = (0, 0)
        #self.torque = 0

        self.speed = 0
        self.state = 0
        if self.Game:
            self.start_game_time = self.Game.game_time
        else:
            self.start_game_time = 0
        self.start_time = time.time()
        self.destroy_countdown = 0


    def get_input_state(self):
        self.input_state = {}
            
        #self.input_state['distances'] = distances

        #TRACK ANGLE AT POSITION :
        # POSITIONS = [5, 10, 20]
        # self.input_state['directions'] = []
        # for pos in POSITIONS:
        #     pos = (self.state + pos) % len(self.Game.Circuit.circuit)
        #     vect = self.Game.Circuit.circuit[pos] - self.position
        #     vect_angle = np.arctan2(vect[1], vect[0])
        #     angle = ((vect_angle - self.angle)/np.pi + 1) % 2 - 1
        #     self.input_state['directions'].append(angle)
        return self.input_state
    
    def get_action(self):
        actions = self.controller.get_action(state=self.input_state)
        actions = np.clip(actions, -1, 1)
        return actions

    # print fuction
    def __str__(self):
        return f'Car : pos={self.position}, angle={self.angle}, state={self.state}'

    def activate(self):
        if not self.is_active:
            self.is_active = True

    def deactivate(self):
        if self.is_active:
            self.is_active = False

