"""
Code to simulate biceps and triceps.
Constant excitation for triceps and using PID for biceps excitation
PID controller works fine if the feedback is Y position of the hand. 
If angle is used as feedback, the controller needs more fine tuning but wokrs fine for target_angle = 204
"""

import sys
import os
# import time
import numpy as np
import matplotlib.pyplot as plt
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from test_single_actuator.envs import PymunkSingleActArmEnv


def main():
    env = PymunkSingleActArmEnv(apply_fatigue=False)

    # Set up the simulation parameters
    sim_duration = 100  # seconds
    frames_per_second = 50
    step_size = 1 / frames_per_second
    total_steps = int(sim_duration / step_size)

    # brachialis_input = np.full(shape=10, fill_value=0) #sending an array of excitation; len(brachialis_input) = motor_unit_count


    def PID_control():
        hand_pos_x = []
        hand_pos_y = []
        lower_arm_angle = []
        error_history = []

        # Here we are going to send a constant excitation to the triceps and
        # vary the excitation of the biceps as we try to hit a target location.
        brachialis_input = 0.5  # Percent of max input
        tricep_input = 0.5
        hand_target_y = 210 #157 to 260

        # Hand tuned PID params.
        Kp = 0.0001
        Ki = 0.01
        Kd = 0.00001
        prev_y = None

        # total_steps = 1000
        for i in range(total_steps):

            hand_x, hand_y, low_arm_angle = env.step([brachialis_input, tricep_input],  step_size, debug=False)
             
            print(low_arm_angle, brachialis_input, tricep_input)

            hand_pos_x.append(hand_x)
            hand_pos_y.append(hand_y)
            lower_arm_angle.append(low_arm_angle)

            # PID Control
            if prev_y is None:
                prev_y = low_arm_angle

            # Proportional component
            error = hand_target_y - low_arm_angle
            alpha = Kp * error
            # Add integral component
            i_c = Ki * (error * step_size)
            alpha -= i_c
            # Add in differential component
            d_c = Kd * ((low_arm_angle - prev_y) / step_size)
            alpha -= d_c

            error_history.append(error)

            prev_y = low_arm_angle
            tricep_input += alpha

            if tricep_input > 5.0:
                tricep_input = 5.0
            if tricep_input < 0.0:
                tricep_input = 0.0

            # print(error, tricep_input)


            env.render()

        time_steps = list(range(total_steps))
        plt.plot(time_steps,lower_arm_angle, label='hand angle')
        plt.plot(time_steps,error_history, label='error')
        plt.legend()
        # plt.ylabel('hand y')
        plt.show()

    PID_control()






    def simple_test():

        brachialis_input = 0.0 #exciting all muscle units by a constant number
        tricep_input = 0.0

        hand_pos_x = []
        hand_pos_y = []
        brach_input_list=[]
        lower_arm_angle = []
    
        for i in range(total_steps):

            # if i%500 == 0:
            #     print("Excitation :", brachialis_input, tricep_input)
            #     if brachialis_input == 0.0: 
            #         brachialis_input = max_exc
            #         tricep_input = 0.0

            #     elif brachialis_input == max_exc: 
            #         brachialis_input = 0.0
            #         tricep_input = max_exc

            
            hand_pos, low_arm_angle = env.step([brachialis_input, tricep_input],  step_size, debug=False)
            hand_x, hand_y = hand_pos

            print(low_arm_angle)

            if i < 2000:
                brachialis_input = 0.5
                tricep_input = 1.1
            elif i > 2000:
                tricep_input = 0.8
                # brachialis_input += 0.001
                brachialis_input = 0.2

            hand_pos_x.append(hand_x)
            hand_pos_y.append(hand_y)
            lower_arm_angle.append(low_arm_angle)
            brach_input_list.append(brachialis_input)

            env.render()
        
        time_steps = list(range(total_steps))
        # plt.plot(time_steps,hand_pos_y, label='hand y position')
        # plt.plot(time_steps,brach_input_list, label='brach input')
        # plt.legend()

        # plt.ylabel('some numbers')
        # plt.show()
       
    # simple_test()




if __name__ == '__main__':
    main()
