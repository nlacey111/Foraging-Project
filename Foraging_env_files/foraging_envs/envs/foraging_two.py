import numpy as np
import gymnasium as gym 
from gymnasium import spaces 
import torch 


''' 
followed this: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''

'''
I am modifying the original foraging environment to include a travel time equivalent. 
This will happen as a disruption to the reward when the agent chooses to leave a flower.
Choose travel time to equal 2 time steps, for now. 

I am also getting rid of the episode_length parameter. Stable baselines handles time steps per episode for us. (TBD)
-> going to try to change this: the episode ends once you have visited 100 flowers. 
'''


class ForagingClass2(gym.Env):
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, episode_length = 100, flower_distribution = "uniform", decay_parameter = .25, travel_time = 2,render_mode = "human"):
        super(ForagingClass2, self).__init__()
            
        self.render_mode = render_mode
        # 2 actions: stay or leave 
        self.action_space = gym.spaces.Discrete(2, start=0) 
        # 0 = stay 
        # 1 = leave
        
        # observation space is what you chose last (0 or 1) and the amount of reward you got (between 0-1) 
        self.observation_space = gym.spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        
        self.iteration = 1
        self.episode_length = episode_length
        self.time_spent = 0 # this is the time spent at the current flower- updates when the agent chooses to stay

        self.decay_parameter = decay_parameter # this is the decay parameter for the exponential decay function
        self.flower_distribution = flower_distribution # this is the distribution of the flowers starting reward 
        self.flower_count = 0 # this is the number of flowers that have been chosen so far

        self.travel_time = travel_time # this is the travel time to the new flower

        self.stay_time_hist_ep = [] # this is the history of the time spent at the current flower- will update every time you leave a flower with "time spent at the flower"
        self.stay_time_hist_all_ep = [] # this is the history of the time spent at all of the flowers- will update when the agent chooses action 1

        self.reward_hist_ep = [] # this is the history of the rewards for the current episode- will update every step 
        self.reward_hist_all_ep = [] # this is the history of the rewards for all episodes- will update when terminated
        # set the flower rewards- first start with they all start at 1 before decaying 
        self.flower_reward = self.find_new_flower_reward() 
        # this will need to be changed when actions are taken- if choose to leave, then a new flower will be chosen 
        
        
    def step(self, action): 
        
        action = int(action) # make sure the action is an integer
        if action == 0: # if choose to stay, the reward should be given based on how long theyve been there. 
            self.time_spent += 1 # increment the time spent at the flower

 
        elif action == 1: # if choose to leave, then a new flower will be chosen
            self.stay_time_hist_ep.append(self.time_spent)
            self.time_spent = 0
            self.flower_reward = self.find_new_flower_reward() # this will be a function that will determine the new flower reward based on the distribution of the flowers
            self.flower_count += 1 # increment the flower count
            
        else: 
            raise ValueError("Action must be 0 or 1. Action that was chosen: ", action)
        
        
        reward = self.find_reward(self.time_spent, self.flower_reward, decay_function = "exponential", choice = action) # this is a function that will determine the reward based on the time spent and the flower reward
        
        self.reward_hist_ep.append(reward) # update the reward history for the current time step
        
        # observation  is the action taken and the reward received- possibly add the time spent at the flower as well
        observation = np.array([action, reward], dtype = np.float32)
        
      
        if self.iteration == self.episode_length:
            terminated = True
            # update all episodic history 
            self.stay_time_hist_all_ep.append(self.stay_time_hist_ep)
            self.reward_hist_all_ep.append(self.reward_hist_ep)
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
        
        # set the flower reward- first start with they all start at 1 before decaying 
        self.flower_reward = self.find_new_flower_reward() 
        # this will need to be changed when actions are taken- if choose to leave, then a new flower will be chosen 
        
        #reset the observation
        observation = np.array([0, 1], dtype = np.float32) # resets to stay and 1 reward 

        # reset local history 
        self.stay_time_hist_ep = [] # this is the history of the time spent at each flower
        self.reward_hist_ep = [] # this is the history of the rewards for the current episode

        # reset the flower count
        self.flower_count = 0 # this is the number of flowers that have been chosen so far
        
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
            
        
    
    def find_reward(self, time_spent, starting_reward, decay_function, choice):
        # this is a function that will determine the reward based on the time spent and the flower reward
        if decay_function == "exponential":
            if choice == 0: # if choose to stay, the reward should be given based on how long theyve been there.
                reward = starting_reward * np.exp(-time_spent*self.decay_parameter)  # this is an exponential decay function with a decay rate of 1/4
                # this is an exponential decay function with a decay rate of 1/4
            else: # if choose to leave, then provide punishment 
                reward = - self.find_c(starting_reward)
        else:
            raise ValueError("Decay function must be exponential. Decay function that was chosen: ", decay_function)
                
        return reward
    
    def find_new_flower_reward(self):
        # this function will determine the new flower reward based on the distribution of the flowers
        
        # self.flower_distribution = "uniform"
        
        new_flower_reward = 1
        return new_flower_reward
    
    def find_c(self, starting_reward): # consider changing t to just be based on last flower visited
        # this function will find the c that is subtracted from the reward 
        t = self.iteration + self.travel_time*(self.flower_count -1) # this is the time spent at the flower
        #  t = self.stay_time_hist_ep[-1]
        c = starting_reward*(t/(t + self.travel_time))
        return c

    