



import gym
import random

random.seed(1234)

streets = gym.make("Taxi-v2").env
streets.render()




initial_state = streets.encode(2, 3, 1, 0) ## R=0, G=1, Y=2 , B=3

streets.s = initial_state

streets.render()




streets.P[initial_state]




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
        
        





q_table[initial_state]





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
    
