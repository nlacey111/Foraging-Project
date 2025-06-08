import numpy as np
import gymnasium as gym 
from gymnasium import spaces 
import torch 


''' 
followed this: https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/
'''

'''
Changing the enviroment to be a tube-like environment where the agent can choose to stay at a flower or leave to find a new one.
The reward will decay exponentially based on the time spent at the flower.
If the agent chooses to leave, then a new flower will be chosen, and the agent will have to travel to the new flower (i.e., choose to leave in all the in between timesteps).

Changing observation space as well to include how far away the next flower is. 

The travel distance is manually set to 5. If changing this to be from a distribution, then make sure to change every instance. 

Need to figure out how to save time spent info from the monitor stuff 
'''


class ForagingClass_tube(gym.Env):
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, episode_length = 5000, flower_distribution = "uniform", decay_parameter = .25, travel_time = 5,render_mode = "human"):
        super(ForagingClass_tube, self).__init__()
            
        self.render_mode = render_mode
        # 2 actions: stay or leave 
        self.action_space = gym.spaces.Discrete(2, start=0) 
        # 0 = stay 
        # 1 = leave

        # to normalize the distance to next flower, we need to divide the distnace by the maximum distance. 
        self.max_distance = 5
        
        # observation space is what you chose last (0 or 1) and the amount of reward you got (between 0-1) 
        # change observation space to include a time until next flower countdown 
        self.observation_space = gym.spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        
        self.iteration = 1
        self.episode_length = episode_length
        self.time_spent = 0 # this is the time spent at the current flower- updates when the agent chooses to stay

        self.decay_parameter = decay_parameter # this is the decay parameter for the exponential decay function
        self.flower_distribution = flower_distribution # this is the distribution of the flowers starting reward 
        self.flower_count = 0 # this is the number of flowers that have been chosen so far

        self.travel_time = travel_time # this is the travel time to the new flower

        # set the flower rewards- first start with they all start at 1 before decaying 
        self.flower_reward = self.find_new_flower_reward() 
        # this will need to be changed when actions are taken- if choose to leave, then a new flower will be chosen 
        
        
    def step(self, action): 
        
        action = int(action) # make sure the action is an integer

        if action == 0: # if choose to stay, the reward should be given based on how long theyve been there. 
            self.time_spent += 1 # increment the time spent at the flower
            if self.travel_time ==5: # if the travel time is 5, then the agent is at the flower
                reward = self.find_reward(self.time_spent, self.flower_reward, decay_function = "exponential", choice = action) # this is a function that will determine the reward based on the time spent and the flower reward
            else: # if the travel time is not 5, then the agent is staying at somewhere that is not a flower 
                reward = 0

        elif action == 1: # if choose to leave, then a new flower will be chosen
            self.time_spent = 0 # reset the time spent at the flower to 0, since they are leaving the flower
            # either they reach a new flower, or they dont reach a new flower 
            if self.travel_time > 1: # if there is a travel time, then the agent will have to wait for the travel time to pass before reaching the new flower
                self.travel_time -= 1 # decrement the travel time
                reward = 0 # no reward for waiting, just waiting for the travel time to pass
             
            elif self.travel_time == 1: # if the travel time is 0, then the agent has reached the new flower
                self.flower_reward = self.find_new_flower_reward() # this will be a function that will determine the new flower reward based on the distribution of the flowers
                self.travel_time = 5 # reset the travel time to 5
                self.flower_count += 1 # increment the flower count
                reward = self.find_reward(self.time_spent, self.flower_reward, decay_function = "exponential", choice = action) # this is a function that will determine the reward based on the time spent and the flower reward
            else: # if the travel time is less than 0, then the agent has reached the new flower
                raise ValueError("Travel time cannot be negative. Travel time that was chosen: ", self.travel_time)
        else: 
            raise ValueError("Action must be 0 or 1. Action that was chosen: ", action)
        

        # observation  is the action taken and the reward received- possibly add the time spent at the flower as well
        observation = np.array([action, reward, self.travel_time/self.max_distance], dtype = np.float32) # the travel time is normalized by the maximum distance, which is 5
        
      
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
        
        # set the flower reward- first start with they all start at 1 before decaying 
        self.flower_reward = self.find_new_flower_reward() 
        # this will need to be changed when actions are taken- if choose to leave, then a new flower will be chosen 
        
        self.travel_time = 5 # this is the travel time to the new flower
        #reset the observation
        observation = np.array([0, 1, self.travel_time/self.max_distance], dtype = np.float32)

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
            reward = starting_reward * np.exp(-time_spent*self.decay_parameter)  # this is an exponential decay function with a decay rate of 1/4
            # this is an exponential decay function with a decay rate of 1/4
        else:
            raise ValueError("Decay function must be exponential. Decay function that was chosen: ", decay_function)
                
        return reward
    
    def find_new_flower_reward(self):
        # this function will determine the new flower reward based on the distribution of the flowers
        
        # self.flower_distribution = "uniform"
        
        new_flower_reward = 1
        return new_flower_reward
    

    