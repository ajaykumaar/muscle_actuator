import sys
import os
# import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import gymnasium as gym
import stable_baselines3
from stable_baselines3 import SAC #PPO, A2C,
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from test_single_actuator.envs import SingleActEnv

# Create log dir
# log_dir = "/tmp/gym/"
# os.makedirs(log_dir, exist_ok=True)

# env = PymunkSingleActArmEnv(apply_fatigue=False)
target_angle = np.deg2rad(210)
env = SingleActEnv(target_angle=target_angle)
# env = Monitor(env, log_dir)
print("action space: ", env.action_space)
print("observation space: ", env.observation_space)

# env = gym.make('Pendulum-v1', g=9.81)
# print("action space: ", env.action_space)
# print("observation space: ", env.observation_space)

# Set up the simulation parameters
sim_duration = 20  # seconds
frames_per_second = 50
step_size = 1 / frames_per_second
total_steps = int((sim_duration / step_size))
print("Step size: ", step_size, "Total steps: ", total_steps)

model = SAC("MlpPolicy", env, train_freq=1, gradient_steps=2, verbose=1)


def evaluate(model, num_steps=1000):
  """
  Evaluate a RL agent
  :param model: (BaseRLModel object) the RL Agent
  :param num_steps: (int) number of timesteps to evaluate it
  :return: (float) Mean reward for the last 100 episodes
  """
  episode_rewards = [0.0]
  lower_arm_angle = []
  reward_list = []
  obs, info = env.reset()
#   print(obs)
  for i in range(num_steps):
      # _states are only useful when using LSTM policies
      action, _states = model.predict(obs)
      # here, action, rewards and dones are arrays
      # because we are using vectorized env
      obs, rewards, terminated, truncated, info = env.step(action)
      lower_arm_angle.append(obs[2])
      reward_list.append(rewards)
      print(action, rewards)

      # Stats
      episode_rewards[-1] += rewards
      if terminated:
          obs, info = env.reset()
          episode_rewards.append(0.0)
  # Compute mean reward for the last 100 episodes
  mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
  print("Mean reward:", mean_100ep_reward, "Num episodes:", len(episode_rewards))

  time_steps = np.arange(0, num_steps*0.02, 0.02)#list(range(sim_duration, step_size))
  print(len(time_steps), len(reward_list))
  plt.figure(0)
  plt.title("Lower Arm Angle")
  plt.plot(time_steps,lower_arm_angle)
  plt.figure(1)
  plt.plot(time_steps,reward_list)
  plt.show()

  return mean_100ep_reward


# model.learn(total_timesteps=10_000)

# evaluate(model)

def test_action_reward(brach, tri):

    obs = env.reset()
    # print(obs)

    lower_arm_angle = []
    reward_list = []
    brachialis_input = brach  # Percent of max input
    tricep_input = tri

    const_action = np.array([brachialis_input, tricep_input]).astype(np.float32)
    # rand_action = env.action_space.sample()

    # print(type(const_action), type(rand_action))
    # print(np.array(const_action).shape, rand_action.shape)

    for i in range(total_steps):
        
        obs, reward, termintated, truncated, info = env.step(const_action, step_size, debug=False)
        lower_arm_angle.append(obs[2])
        reward_list.append(reward)
        # print(reward)

        # tricep_input += 0.01

        # print(obs[2])

        env.render()
        

        # print(obs[2]-target_angle, reward)

    # print("current obs: ", obs)
    # obs = env.reset()
    # print(obs)
    print(np.abs(lower_arm_angle[-1] - target_angle))
    time_steps = np.arange(0, sim_duration, step_size)#list(range(sim_duration, step_size))
    plt.figure(0)
    plt.title("Lower Arm Angle")
    plt.plot(time_steps,lower_arm_angle)
    plt.figure(1)
    plt.title("Reward")
    plt.plot(time_steps,reward_list)
    plt.show()

test_action_reward(brach=1.0, tri=2)



