import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
import os 
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy
import gymnasium as gym
import foraging_envs
from torch import nn
from convert_net import *
from stable_baselines3 import DQN
import os

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results,  ts2xy

from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import datetime
from stable_baselines3 import PPO



# network_copy_args = dict(obs_space = env.observation_space, action_space =env.action_space, net_arch = net_arch, activation_fn = activation_fn)


class CustomNetwork(nn.Module):
    def __init__(self, network_args = None ):
        # Make network
        super(CustomNetwork, self).__init__()

        self.net_arch = network_args['net_arch']
        self.activation_fn = network_args['activation_fn'] 
        
        self.obs_space = network_args['obs_space']
        self.action_space = network_args['action_space']

        self.mod_list = self.make_modlist()

        self.net = nn.Sequential(
            nn.Linear(self.obs_space, self.net_arch[0]),
            nn.ReLU(),
            *self.mod_list,  # Unpack the module list
            nn.Linear(self.net_arch[-1], self.action_space),  # Assuming the last layer of mod_list is connected to this
        )

    def forward(self, x):
        # Process and give me activations
        activations = []

        for i,l in enumerate(self.net):
            x = self.net[i//2](x) + l(x)
            activations.append(x)

        return activations
    
    def make_modlist(self):
        module_list = []
        for i in range(len(self.net_arch)-1):
            module_list.append(nn.Linear(self.net_arch[i], self.net_arch[i+1]))
            module_list.append(nn.ReLU())
        return module_list

# net1 is original, net2 is the one to copy to  
def copy_weights(from_net, to_net):
    copied_params = []

    for key in from_net.get_parameters()['policy'].keys():
        if "q_net_target" in key:
            copied_params.append(from_net.get_parameters()['policy'][key])

    params2 = list(to_net.parameters())
    for i in range(len(params2)):
        params2[i].requires_grad = False  # Disable gradient updates for the copied parameters
        params2[i].copy_(copied_params[i])
        params2[i].requires_grad = True  # Re-enable gradient updates
    return to_net
    
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
    
def create_model(env_dict, model_type, net_arch, activation_fn, total_timesteps):
    
    env = gym.make(env_dict['env_name'], 
                   episode_length = env_dict["episode_length"], 
                   flower_distribution = env_dict["flower_distribution"],
                   decay_parameter = env_dict["decay_parameter"],
                   travel_time = env_dict["travel_time"],
                   render_mode = env_dict["render_mode"])

    # make sure the environment is valid
    check_env(env, warn=True)
             
    # create folder to put everything in 
    # string made from the network type (DQN, PPO, etc), the environment, and the date
    #create the things to make the folder structure and name
    env_name = env.spec.id
    env_name = env_name.replace("foraging_envs/", "")
    env_name = env_name.replace("-", "_")

    # create the model, then can use info from it to make the file path 
    policy_kwargs = dict(net_arch = net_arch, 
                         activation_fn = activation_fn)
    if model_type == "DQN":
        model = DQN("MlpPolicy", env, verbose=1, policy_kwargs = policy_kwargs)
    elif model_type == "PPO":
        model = PPO("MlpPolicy", env, verbose=1, policy_kwargs = policy_kwargs)
    else:
        raise ValueError(f"Model type {model_type} not recognized. Use 'DQN' or 'PPO'.")
    
    model_type = str(model.policy._get_name()) # will be DQNPolicy or PPOPolicy, etc.

    log_dur = f"Saved Models/{env_name}/{model_type}_{str(datetime.datetime.now())[:10]}/"
    if not os.path.exists(log_dur):
        os.makedirs(log_dur)
    print(f"Logging to {log_dur}")

    # make the env monitor go to this folder
    # wrap env in Monitor to log rewards and other info
    env = Monitor(env, log_dur)




    # choose callback function. this will save the best model based on the mean reward
    # and also evaluate the model every 1000 steps
    callback = EvalCallback(env, best_model_save_path=log_dur, log_path=log_dur, eval_freq=1000, deterministic=True, n_eval_episodes=5)
    model.learn(total_timesteps=total_timesteps, progress_bar=False, log_interval=4, callback = callback)

    # print a string so I can load the best model later 
    print(f"Model saved to {log_dur}best_model.zip")
    # return the model and the log directory
    
    # Create the network copy 
    state, info = env.reset()
    n_observations = len(state)
    network_copy_args = dict(obs_space = n_observations, action_space =env.action_space.n, net_arch = net_arch, activation_fn = activation_fn)
    network_copy = CustomNetwork(network_copy_args)
    network_copy = copy_weights(from_net=model, to_net = network_copy)

    print(f"Network copy created with architecture: {network_copy_args['net_arch']} and activation function: {network_copy_args['activation_fn'].__name__}")
    
    # save the network copy to the log directory
    