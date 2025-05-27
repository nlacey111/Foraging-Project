import numpy as np
import gymnasium as gym 
from gymnasium import spaces 
import torch 


''' 
followed this: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''

'''
This is a foraging environment where the agent has to choose between staying at a flower or leaving it.
The agent receives a reward based on the time spent at the flower and the starting reward of the flower.
The starting reward is currently set to 1, but can be changed to be drawn from a distribution of choice (needs implementation)
Each time step that the agent stays at the flower, the reward decays exponentially based on the time spent at the flower and a decay parameter.
The agent can choose to leave the flower at any time, which will reset the time spent at the flower and give a new starting reward.
'''


class ForagingClass(gym.Env):
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, episode_length = 100, flower_distribution = "uniform", decay_parameter = .25, render_mode = "human"):
        super(ForagingClass, self).__init__()
            
        self.render_mode = render_mode
        # 2 actions: stay or leave 
        self.action_space = gym.spaces.Discrete(2, start=0) 
        # 0 = stay 
        # 1 = leave
        
        # observation space is what you chose last (0 or 1) and the amount of reward you got (between 0-1) 
        self.observation_space = gym.spaces.Box(0, 1, shape=(2,), dtype=np.float32)
        
        self.iteration = 1
        self.episode_length = episode_length
        self.time_spent = 0 # this is the time spent at the current flower
        self.decay_parameter = decay_parameter # this is the decay parameter for the exponential decay function
        self.flower_distribution = flower_distribution # this is the distribution of the flowers starting reward 
        self.flower_count = 0 # this is the number of flowers that have been chosen so far
        
        # set the flower rewards- first start with they all start at 1 before decaying 
        self.flower_reward = self.find_new_flower_reward() 
        # this will need to be changed when actions are taken- if choose to leave, then a new flower will be chosen 
        
        
    def step(self, action): 
        
        action = int(action) # make sure the action is an integer
        if action == 0: # if choose to stay, the reward should be given based on how long theyve been there. 
            self.time_spent += 1 # increment the time spent at the flower
 
        elif action == 1: # if choose to leave, then a new flower will be chosen
            self.time_spent = 0
            self.flower_reward = self.find_new_flower_reward() # this will be a function that will determine the new flower reward based on the distribution of the flowers
            self.flower_count += 1 # increment the flower count
            
        else: 
            raise ValueError("Action must be 0 or 1. Action that was chosen: ", action)
        
        
        reward = self.find_reward(self.time_spent, self.flower_reward, decay_function = "exponential") # this is a function that will determine the reward based on the time spent and the flower reward
        
        # observation  is the action taken and the reward received- possibly add the time spent at the flower as well
        observation = np.array([action, reward], dtype = np.float32)
        
      
        if self.iteration == self.episode_length:
            terminated = True
        else:
            terminated = False
        
        self.iteration = self.iteration + 1
        
        # info is an empty dictionary, DO NOT WRITE AS SELF.INFO = {} because it will overwrite the class variable
        info = {}
        
        # not using truncated
        truncated = False
        
        # return observation, reward, done
        return  observation, reward, terminated, truncated, info # must be in this order, with these names
    
  
    def reset(self, seed=None, options={}):
        # reset the iteration
        self.iteration = 1
        
        self.time_spent = 0 # this is the time spent at the current flower

        self.flower_distribution = self.flower_distribution # this is the distribution of the flowers starting reward 
        
        # set the flower rewards- first start with they all start at 1 before decaying 
        self.flower_reward = self.find_new_flower_reward() 
        # this will need to be changed when actions are taken- if choose to leave, then a new flower will be chosen 
        
        #reset the observation
        observation = np.array([0, 1], dtype = np.float32) # resets to stay and 1 reward 
        
        return observation, {} # must be in this order, with these names
    
    def render(self, mode= "human"):
        
        # show a row of dots representing the flowers
        # the color of the dots will represent the reward of the flower (i.e., how long they have been there for now) 
        # put a number of black dots for the flowers that have not been chosen yet
        # once i keep track of the rewards and how long they have been there, then i can change the color of the dots to represent the reward of the flower (i.e., how long they have been there for now)
        print("Rendering the environment")
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        string_to_print = ""
        for i in range(self.episode_length): 
            if i < self.flower_count:
                string_to_print += GREEN + "x" # green x for the flowers that have been chosen
                # reset color 
                string_to_print += '\033[0m'
            else:
                string_to_print += YELLOW + "o"
                # reset color
                string_to_print += '\033[0m'
        print(string_to_print) # print the string of dots
            
        
    
    def find_reward(self, time_spent, starting_reward, decay_function):
        # this is a function that will determine the reward based on the time spent and the flower reward
        if decay_function == "exponential":
            # this is an exponential decay function
            reward = starting_reward * np.exp(-time_spent*self.decay_parameter) # this is an exponential decay function with a decay rate of 1/4
        else:
            raise ValueError("Decay function must be exponential. Decay function that was chosen: ", decay_function)
                
        return reward
    
    def find_new_flower_reward(self):
        # this function will determine the new flower reward based on the distribution of the flowers
        
        # self.flower_distribution = "uniform"
        
        new_flower_reward = 1
        return new_flower_reward
    