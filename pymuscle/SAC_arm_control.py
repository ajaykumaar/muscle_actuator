import sys
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
import stable_baselines3
from stable_baselines3 import SAC #PPO, A2C,
# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from typing import Callable

from test_single_actuator.envs import SingleActEnv

log_dir = "pymuscle/saved_models/"
target_angle = 210
env = SingleActEnv(target_angle=target_angle)
# env = Monitor(env, log_dir) #comment out this line before evaluation
print("action space: ", env.action_space)
print("observation space: ", env.observation_space)

# Set up the simulation parameters
sim_duration = 20  # seconds
frames_per_second = 50
step_size = 1 / frames_per_second
total_steps = int((sim_duration / step_size))
print("Step size: ", step_size, "Total steps: ", total_steps)

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """

    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}"
                    )

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}.zip")
                    self.model.save(self.save_path)

        return True

def evaluate_model(model, env, num_steps = 1000, save_results = True):
    
    frames_list = []
    actions_list = []
    lower_arm_angle = []
    reward_list = []
    obs, info = env.reset()

    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, info = env.step(action)
        reward_list.append(rewards)
        lower_arm_angle.append(obs[2])
        actions_list.append(action)
        frame = env.render()
        frames_list.append(frame)

        if terminated:
            obs,info = env.reset()

    if save_results == True:
        save_path = "pymuscle/simulation_results/"
        #save video
        if len(frames_list) > 0:
            # choose codec according to format needed
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            video = cv2.VideoWriter(save_path + "simulation.avi", fourcc, 200, (600, 600))

            for frame in frames_list:
                video.write(frame)
            cv2.destroyAllWindows()
            video.release()

        else:
            print("frames list is empty")
    actions_list = np.array(actions_list)
    time_steps = np.arange(0, num_steps*0.02, 0.02)
    plt.figure(0)
    plt.title("Lower arm angle")
    plt.plot(time_steps, lower_arm_angle)
    if save_results == True: plt.savefig(save_path + "lower_arm_angle.png")
    plt.figure(1)
    plt.title("Rewards")
    plt.plot(time_steps, reward_list)
    if save_results == True: plt.savefig(save_path + "rewards.png")
    plt.figure(2)
    plt.title("Muscle excitation")
    plt.plot(time_steps, actions_list[:,0], label = "biceps")
    plt.plot(time_steps, actions_list[:,1], label = "tricepts")
    plt.legend()
    if save_results == True: plt.savefig(save_path + "muscle_excitation.png")
    plt.show()


def linear_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.
        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


#### MAIN #######

model = SAC("MlpPolicy", env, train_freq=1, learning_rate= linear_schedule(0.004), gradient_steps=2, verbose=1)

#load saved agent
saved_model_path = "pymuscle/saved_models/best_model_og.zip"
saved_model = model.load(path= saved_model_path)

# Train the agent
# callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

# model.learn(total_timesteps=30_000, callback = callback)
target_angle = 210
env = SingleActEnv(target_angle= target_angle, potvin_chart= False)
evaluate_model(saved_model, env,num_steps=1500, save_results = True)
# env.create_potvin_chart()

