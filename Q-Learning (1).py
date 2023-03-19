#!/usr/bin/env python
# coding: utf-8

# # Reinforcement Learning
# 
# Let's describe the "taxi problem". We want to build a self-driving taxi that can pick up passengers at one of a set of fixed locations, drop them off at another location, and get there in the quickest amount of time while avoiding obstacles.
# 
# The AI Gym lets us create this environment quickly: 

# In[37]:


import gym
import random

random.seed(1234)

streets = gym.make("Taxi-v2").env
streets.render()


# Let's break down what we're seeing here:
# 
# -  R, G, B, and Y are pickup or dropoff locations.
# -  The BLUE letter indicates where we need to pick someone up from.
# -  The MAGENTA letter indicates where that passenger wants to go to.
# -  The solid lines represent walls that the taxi cannot cross.
# -  The filled rectangle represents the taxi itself - it's yellow when empty, and green when carrying a passenger.

# Our little world here, which we've called "streets", is a 5x5 grid. The state of this world at any time can be defined by:
# 
# -  Where the taxi is (one of 5x5 = 25 locations)
# -  What the current destination is (4 possibilities)
# -  Where the passenger is (5 possibilities: at one of the destinations, or inside the taxi)
# 
# So there are a total of 25 x 4 x 5 = 500 possible states that describe our world.
# 
# For each state, there are six possible actions:
# 
# -  Move South, East, North, or West
# -  Pickup a passenger
# -  Drop off a passenger
# 
# Q-Learning will take place using the following rewards and penalties at each state:
# 
# -  A successfull drop-off yields +20 points
# -  Every time step taken while driving a passenger yields a -1 point penalty
# -  Picking up or dropping off at an illegal location yields a -10 point penalty
# 
# Moving across a wall just isn't allowed at all.
# 
# Let's define an initial state, with the taxi at location (2, 3), the passenger at pickup location 2, and the destination at location 0:

# In[78]:


initial_state = streets.encode(2, 3, 1, 0) ## R=0, G=1, Y=2 , B=3

streets.s = initial_state

streets.render()


# Let's examine the reward table for this initial state:

# In[79]:


streets.P[initial_state]


# Here's how to interpret this - each row corresponds to a potential action at this state: move South, North, East, or West, pickup, or dropoff. The four values in each row are the probability assigned to that action, the next state that results from that action, the reward for that action, and whether that action indicates a successful dropoff took place. 
# 
# So for example, moving North from this state would put us into state number 368, incur a penalty of -1 for taking up time, and does not result in a successful dropoff.
# 
# So, let's do Q-learning! First we need to train our model. At a high level, we'll train over 10,000 simulated taxi runs. For each run, we'll step through time, with a 10% chance at each step of making a random, exploratory step instead of using the learned Q values to guide our actions.

# In[80]:


import numpy as np

q_table = np.zeros([streets.observation_space.n, streets.action_space.n])

learning_rate = 0.1
discount_factor = 0.6
exploration = 0.1  # PROBABILITY EXPLORATION
epochs = 10000   #EPOCHS STEPS

for taxi_run in range(epochs):
    state = streets.reset()
    done = False
    
    while not done:
        random_value = random.uniform(0, 1)
        if (random_value < exploration):
            action = streets.action_space.sample() # Explore a random action
        else:
            action = np.argmax(q_table[state]) # Use the action with the highest q-value
            
        next_state, reward, done, info = streets.step(action)
        
        prev_q = q_table[state, action]
        next_max_q = np.max(q_table[next_state])
        new_q = (1 - learning_rate) * prev_q + learning_rate * (reward + discount_factor * next_max_q) # FORMULA OF Q LEARNING
        q_table[state, action] = new_q
        
        state = next_state
        
        


# So now we have a table of Q-values that can be quickly used to determine the optimal next step for any given state! Let's check the table for our initial state above:

# In[81]:


q_table[initial_state]


# The lowest q-value here corresponds to the action "go West", which makes sense - that's the most direct route toward our destination from that point. It seems to work! Let's see it in action!

# In[83]:


from IPython.display import clear_output
from time import sleep

for tripnum in range(1, 10):
    state = streets.reset()
   
    done = False
    
    while not done:
        action = np.argmax(q_table[state])
        next_state, reward, done, info = streets.step(action)
        clear_output(wait=True)
        print("Trip number " + str(tripnum))
        print(streets.render(mode='ansi'))
        sleep(.5)
        state = next_state
        
  #  sleep(2)
    


# ## Your Challenge
# 
# Modify the block above to keep track of the total time steps, and use that as a metric as to how good our Q-learning system is. You might want to increase the number of simulated trips, and remove the sleep() calls to allow you to run over more samples.
# 
# Now, try experimenting with the hyperparameters. How low can the number of epochs go before our model starts to suffer? Can you come up with better learning rates, discount factors, or exploration factors to make the training more efficient? The exploration vs. exploitation rate in particular is interesting to experiment with.

# In[ ]:





# In[ ]:




