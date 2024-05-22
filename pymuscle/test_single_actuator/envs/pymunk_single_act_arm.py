import sys
# import gym
import gymnasium as gym
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE)
# from pymuscle import StandardMuscle as Muscle
from pymuscle import PotvinFuglevandMuscle as Muscle
#Uncomment the below line to run on google colab
# from pymuscle.pymuscle import PotvinFuglevandMuscle as Muscle 


# from pymunk.constraints import *

class PymunkSingleActArmEnv(gym.Env):
    """
    Single joint arm with opposing muscles physically simulated by Pymunk in
    a Pygame wrapper.
    """

    def __init__(self, apply_fatigue=False):
        # Set up our 2D physics simulation
        self._init_sim()

        # Add a simulated arm consisting of:
        #  - bones (rigid bodies)
        #  - muscle bodies (damped spring constraints)
        self.brach, self.tricep = self._add_arm()
        # self.brach = self._add_arm()

        # Instantiate the PyMuscles
        self.brach_muscle = Muscle(
            motor_unit_count=10,
            apply_peripheral_fatigue= False  #apply_fatigue
        )
        self.tricep_muscle = Muscle(
            motor_unit_count=10,
            apply_peripheral_fatigue=False  # Tricep never gets tired in this env
        )

        self.frames = 0

    def _init_sim(self):
        pygame.init()
        screen_width = screen_height = 600
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Curl Sim")
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = 1  # Disable constraint drawing
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -980.0)
        # self.copy_lower_arm = None ####### custom

    def _add_arm(self):
        # config = {
        #     "arm_center": (self.screen.get_width() / 2,
        #                    self.screen.get_height() / 2),
        #     "lower_arm_length": 170,
        #     "lower_arm_starting_angle": 90,
        #     "lower_arm_mass": 10,
        #     "brach_rest_length": 5,
        #     "brach_stiffness": 450,
        #     "brach_damping": 200,
        #     "tricep_rest_length": 30,
        #     "tricep_stiffness": 50,
        #     "tricep_damping": 400
        # }

        #custom similar springs for brach and triceps
        config = {
            "arm_center": (self.screen.get_width() / 2,
                           self.screen.get_height() / 2),
            "lower_arm_length": 170,
            "lower_arm_starting_angle": 140,
            "lower_arm_mass": 10,
            "brach_rest_length": 5,
            "brach_stiffness": 450,
            "brach_damping": 200,
            "tricep_rest_length": 30,
            "tricep_stiffness": 50,
            "tricep_damping": 400
        }
        
        # Upper Arm
        upper_arm_length = 200
        upper_arm_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        upper_arm_body.position = config["arm_center"]
        upper_arm_body.angle = np.deg2rad(120)
        upper_arm_line = pymunk.Segment(upper_arm_body, (0, 0), (-upper_arm_length, 0), 5)
        upper_arm_line.sensor = True  # Disable collision

        self.space.add(upper_arm_body)
        self.space.add(upper_arm_line)

        # Lower Arm
        lower_arm_body = pymunk.Body(0, 0)  # Pymunk will calculate moment based on mass of attached shape
        lower_arm_body.position = config["arm_center"]
        lower_arm_body.angle = np.deg2rad(config["lower_arm_starting_angle"])
        elbow_extension_length = 20
        lower_arm_start = (-elbow_extension_length, 0)
        lower_arm_line = pymunk.Segment(
            lower_arm_body,
            lower_arm_start,
            (config["lower_arm_length"], 0),
            5
        )
        lower_arm_line.mass = config["lower_arm_mass"]
        lower_arm_line.friction = 1.0

        self.space.add(lower_arm_body)
        self.space.add(lower_arm_line)

        self.copy_lower_arm = lower_arm_body ###### custom

        # Hand
        hand_width = hand_height = 15
        start_x = config["lower_arm_length"]
        start_y = 14
        self.hand_shape = pymunk.Circle(
            lower_arm_body,
            20,
            (start_x, start_y)
        )
        # self.hand_shape.mass = 0.1
        self.space.add(self.hand_shape)

        # Pivot (Elbow)
        elbow_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        elbow_body.position = config["arm_center"]
        elbow_joint = pymunk.PivotJoint(elbow_body, lower_arm_body, config["arm_center"])
        self.space.add(elbow_joint)

        # Spring (Brachialis Muscle)
        brach_spring = pymunk.DampedSpring(
            upper_arm_body,
            lower_arm_body,
            (-(upper_arm_length * (1 / 2)), 0),  # Connect half way up the upper arm
            (config["lower_arm_length"] / 4, 0),  # Connect near the bottom of the lower arm
            config["brach_rest_length"],
            config["brach_stiffness"],
            config["brach_damping"]
        )
        self.space.add(brach_spring)

        # Spring (Tricep Muscle)

        
        tricep_spring = pymunk.DampedSpring(
            upper_arm_body,
            lower_arm_body,
            (-(upper_arm_length * (3 / 4)), 0),
            lower_arm_start,
            config["tricep_rest_length"],
            config["tricep_stiffness"],
            config["tricep_damping"]
        )
        self.space.add(tricep_spring)

        # Elbow stop (prevent under/over extension)
        elbow_stop_point = pymunk.Circle(
            upper_arm_body,
            radius=5,
            offset=(-elbow_extension_length, -3)
        )
        elbow_stop_point.friction = 1.0
        self.space.add(elbow_stop_point)

        return brach_spring, tricep_spring

    @staticmethod
    def _handle_keys():
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit(0)
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                sys.exit(0)

    def step(self, input_array, step_size, debug=True):
        # Check for user input
        self._handle_keys()

        # Scale input to match the expected range of the muscle sim
        # input_array = np.array(input_array)

        if debug:
            print(input_array)

        # Advance the simulation
        self.space.step(step_size)
        self.frames += 1

        # Advance muscle sim and sync with physics sim
        brach_output = self.brach_muscle.step(input_array[0], step_size)
        # print("current force: ", self.brach_muscle.current_forces)
        # brach_output = self.brach_muscle.step(input_array, step_size)
        tricep_output = self.tricep_muscle.step(input_array[1], step_size)
        # tricep_output = self.tricep_muscle.step(input_array, step_size)

        gain = 500
        self.brach.stiffness = brach_output * gain
        self.tricep.stiffness = tricep_output * gain

        if debug:
            print("Brach Total Output: ", brach_output)
            print("Tricep Total Output: ", tricep_output)
            print("Brach Stiffness: ", self.brach.stiffness)
            print("Tricep Stiffness: ", self.tricep.stiffness)

        hand_location = self.hand_shape.body.local_to_world((170, 0))
        return hand_location[0], hand_location[1] , np.rad2deg(self.copy_lower_arm.angle)

    def render(self, debug=True):
        if debug and (self.draw_options.flags != 3):
            self.draw_options.flags = 3  # Enable constraint drawing

        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()

    def reset(self):
        if self.space:
            del self.space
        self._init_sim()


class SingleActEnv(PymunkSingleActArmEnv):

    def __init__(self, action_size=2, step_size = 0.002, target_angle= 210):
        super().__init__()

        self.action_space = gym.spaces.Box(low=0.0, high=2.0, shape=(action_size,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.current_time = 0.0
        self.actions = None
        self.step_size = step_size

        self.target_angle = target_angle
        self.prev_angle = None
        self.prev_actions = None

    def step(self, actions, debug=False):
        # Check for user input
        self._handle_keys()
        self.actions = actions

        if debug:
            print(self.actions)
            print(self.actions[0], self.actions[1])

        self.current_time += self.step_size
        #converting float32 action to float64 bcause pymuscle's muscle step only takes float64
        self.actions = self.actions.astype(np.float64)

        self.space.step(self.step_size)
        self.frames += 1

        # Advance muscle sim and sync with physics sim
        brach_output = self.brach_muscle.step(self.actions[0], self.step_size)
        tricep_output = self.tricep_muscle.step(self.actions[1], self.step_size)

        gain = 500
        self.brach.stiffness = brach_output * gain
        self.tricep.stiffness = tricep_output * gain

        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_done()
        truncated = False
        info = {}
        # print("obs from inside step :", obs)
        #update previous action after reward calculation
        self.prev_actions = actions

        return obs, reward, terminated, truncated, info  
    
    def reset(self, seed= None):
        self._init_sim()
        self.brach, self.tricep = self._add_arm()
        # self.brach = self._add_arm()

        # Instantiate the PyMuscles
        self.brach_muscle = Muscle(
            motor_unit_count=10,
            apply_peripheral_fatigue=False
        )
        self.tricep_muscle = Muscle(
            motor_unit_count=10,
            apply_peripheral_fatigue=False  # Tricep never gets tired in this env
        )
        obs = self._get_observation()
        info = {}

        return obs, info

    def _get_observation(self):

        # hand_location = self.hand_shape.body.local_to_world((170, 0))
        # hand_x = hand_location[0]
        # hand_y = hand_location[1]

        arm_angle = np.rad2deg(self.copy_lower_arm.angle)

        hand_x = np.cos(self.copy_lower_arm.angle)
        hand_y = np.sin(self.copy_lower_arm.angle)
        # print(self.space.current_time_step)

        if self.prev_angle is None:
            self.prev_angle = arm_angle
            theta_dt = 0
        elif self.prev_angle is not None:
            theta_dt = (np.abs(arm_angle - self.prev_angle))/self.step_size
            self.prev_angle = arm_angle

        # print("Arm coordinate check: ", [hand_x, hand_y], [np.cos(arm_angle), np.sin(arm_angle)])

        return np.array([hand_x, hand_y, arm_angle]).astype(np.float32)
    
    def _is_done(self):
        done = False
        
        # if the lower arm angle goes beyond the range [120,260] the episode ends
        arm_angle = np.rad2deg(self.copy_lower_arm.angle)

        if arm_angle > 260 or arm_angle < 120:
            done = True
        elif arm_angle == self.target_angle:
            done = True

        #if the lower arm angle hasn't reached the target in 10 seconds the episode ends
        # if self.current_time > 10:
        #     if np.isclose(self.target_angle, arm_angle, atol= 5) == 0:
        #         done = True

        return done
    
    def _get_reward(self):
        ### using reward function from pendulum-v1 env
        current_angle = self._get_observation()[2] #current_angle
        error = np.abs(current_angle - self.target_angle)

        if self.prev_angle is None:
            self.prev_angle = current_angle
            theta_dt = 0

        elif self.prev_angle is not None:
            theta_dt = (np.abs(current_angle - self.prev_angle))/self.step_size
            self.prev_angle = current_angle

        if self.prev_actions is None:
            br_exi_dt = 0
            tr_exi_dt = 0
        
        elif self.prev_actions is not None:
            br_exi_dt = (np.abs(self.actions[0] - self.prev_actions[0]))/self.step_size
            tr_exi_dt = (np.abs(self.actions[1] - self.prev_actions[1]))/self.step_size


        # print(error, theta_dt)
        # reward = -(0.1*(error**2)) #+ 0.01*(theta_dt**2)) #* (-0.1*self.current_time)

        reward = -(error**2) #+ 0.1*(theta_dt**2)) #+ 0.1*(br_exi_dt + tr_exi_dt)) 
        # print("reward: ", reward)


        ##### custom reward
        #reward = m*|current_angle - target_angle| + c*time   ; m= -1, c=100
        #max possible reward is c = 500, if target angle is reached

        # current_angle = self._get_observation()[2] #current_angle
        # difference = np.abs(current_angle - self.target_angle)

        # if current_angle == self.target_angle:
        #     reward = 500
        # else:
        #     reward = -self.current_time*(difference) + (10*self.current_time)

        return reward




    #Using DQN is a bad idea because I'd have to approximate the action space. The problem is more like approximating the system,
    #i.e., given (s,a) what is the qvalue, rather than explointing the action space of a muscle actuator. 
    # a preferred approach would be an actor-critic solution because it directly maps states to actions hence might be better suited
    #for a muscle actuator




    
        

