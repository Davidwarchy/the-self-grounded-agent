For action module, modify .step (or other places) to store this information
step
strategy 
action 
run_start (1 or blank; to indicate if current step is the start of a run) 
run_length (the length of the run, only for run start step) 
lidar readings as ray_0,..., ray_99 (for 100 rays) 
episode (what episode we are in)
reward (option to pass a reward in the action module)
x
y 
z 

for each row: the step is step i, the action is the action taken at step i, the lidar readings at step i are the readings after action i, the reward is after taking action i... x,y,orientation aslo after action

What's the best strategy for adding reward to a step? 

## Strategies 
So for each strategy, we can have columns to store for storable data for each step - ie, if current step is run start and the run length, the remaining steps, ... and specify whether these should be stored by robot_env at each step... robot env can query strategy to see what it should store, then store them.
uniform - run_start, steps_remaining 
levy - run_start, steps_remaining 
random_walk - no extra info 
manual - no extra info 

## Action
For the rl 
- We have a calculate reward functino in the base robot env
- We will override it with our specific reaward function 

## Base Robot Env step function 
- get etra info from each strategy 
- if we have an active reward, get the reward after stepping (so we want to have a condition to check if we have active reward) 


Am I right here? If so, Let's implement the necessary changes