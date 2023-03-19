# Self_Driving_Car
Made an AI based self-driving taxi that can pick up passengers at one of a set of fixed locations, drop them off at another location, and get there in the quickest amount of time while avoiding obstacles using a simple python code.
## REINFORCEMENT LEARNING
The AI Gym  create this environment :

![GYM 1](https://user-images.githubusercontent.com/43465317/226190048-2a705156-5148-4dd6-be1d-f4df9fd8c412.png)

Let's break down what we're seeing here:

R, G, B, and Y are pickup or dropoff locations.
The BLUE letter indicates where we need to pick someone up from.
The MAGENTA letter indicates where that passenger wants to go to.
The solid lines represent walls that the taxi cannot cross.
The filled rectangle represents the taxi itself - it's yellow when empty, and green when carrying a passenger.
Our little world here, which we've called "streets", is a 5x5 grid. The state of this world at any time can be defined by:

Where the taxi is (one of 5x5 = 25 locations)
What the current destination is (4 possibilities)
Where the passenger is (5 possibilities: at one of the destinations, or inside the taxi)
So there are a total of 25 x 4 x 5 = 500 possible states that describe our world.

For each state, there are six possible actions:

Move South, East, North, or West
Pickup a passenger
Drop off a passenger
Q-Learning will take place using the following rewards and penalties at each state:

A successfull drop-off yields +20 points
Every time step taken while driving a passenger yields a -1 point penalty
Picking up or dropping off at an illegal location yields a -10 point penalty
Moving across a wall just isn't allowed at all.

Let's define an initial state, with the taxi at location (2, 3), the passenger at pickup location 2, and the destination at location 0:

![GYM 2](https://user-images.githubusercontent.com/43465317/226190182-a8af39f8-7cbc-49ac-9882-9b8c5e13a8ed.png)

#Let's examine the reward table for this initial state:

![GYM 3](https://user-images.githubusercontent.com/43465317/226190264-c305ba8a-01d0-4c76-bb96-876a161ccac6.png)

Here's how to interpret this - each row corresponds to a potential action at this state: move South, North, East, or West, pickup, or dropoff. The four values in each row are the probability assigned to that action, the next state that results from that action, the reward for that action, and whether that action indicates a successful dropoff took place.

So for example, moving North from this state would put us into state number 368, incur a penalty of -1 for taking up time, and does not result in a successful dropoff.

# Q LEARNING

First we need to train our model. At a high level, we'll train over 10,000 simulated taxi runs. For each run, we'll step through time, with a 10% chance at each step of making a random, exploratory step instead of using the learned Q values to guide our actions.

## Q TABLE

![GYM 4](https://user-images.githubusercontent.com/43465317/226190332-0210c371-e51e-4a81-bcb0-f641f2dbe021.png)

# HEREIS THE ULTIMATE RESULT 

After 10 times trip.

![GYM 5](https://user-images.githubusercontent.com/43465317/226190966-3eb3a2a0-77d6-427f-979c-541e1507e92a.png)
