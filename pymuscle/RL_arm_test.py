import sys
import os
# import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from test_single_actuator.envs import SingleActEnv

import stable_baselines3
from stable_baselines3 import SAC #PPO, A2C,
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# env = PymunkSingleActArmEnv(apply_fatigue=False)
target_angle = 204
env = SingleActEnv(target_angle=target_angle)
print("action space: ", env.action_space)
print("observation space: ", env.observation_space)
# env = Monitor(env)

# model = SAC("MlpPolicy", env, train_freq=1, gradient_steps=2, verbose=1)

# model.learn(total_timesteps=100)



# Set up the simulation parameters
sim_duration = 100  # seconds
frames_per_second = 50
step_size = 1 / frames_per_second
total_steps = 1500 #int(sim_duration / step_size)

brachialis_input = 1.0  # Percent of max input
tricep_input = 2.0

obs = env.reset()
print(obs)

lower_arm_angle = []

for i in range(total_steps):
    
    obs, reward, done, info = env.step([brachialis_input, tricep_input],  step_size, debug=False)
    lower_arm_angle.append(obs[2])

    # tricep_input += 0.01

    # print(obs[2])

    env.render()

    # print(obs[2]-target_angle, reward)

print("current obs: ", obs)
obs = env.reset()
print(obs)
time_steps = list(range(total_steps))
plt.plot(time_steps,lower_arm_angle)
plt.show()

# for i in range(10):

#     hand_x, hand_y, low_arm_angle = env.step([brachialis_input, tricep_input],  step_size, debug=False)

#     print(low_arm_angle, hand_x, hand_y)

