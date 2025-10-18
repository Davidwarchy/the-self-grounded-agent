# Physical Description 
## Environment 
Room with obstacles of various shapes 
## Robot 
Body: Cylinder with two wheels that are differentially driven and lidar
Sensors: lidar; 1 layer; 100 rays
Actuators: 2 wheels (capable of differential steering) 

# Goal
## Description  
* To be able to move to perceptions previously encountered
* I've previously been able to sort of relate a topology to sensor readings; but that's not what I want to do right now. 
* I think this seems the thing closest to making robots autonomous (as opposed to making robots learn 3d maps of the environment) 
* To amke robots truly autonomous & useful for human related activities, we can have teh robots be programmable (to achieve certain tasks); but this is a secondary thing (when it comes to true autonomy). This usefulness to human beings is a special class of problem. 

## Challenges
* We might not know exactly how a scene should look from all perspectives (from angles and distances). We only have a tiny perspective. 

# Exploration
* We move randomly in an environment
* Actions are discretized: left, right, up, down (it's a big problem to determine the right kinds of steps) 
* We record **lidar readings**, in sequence; we also record **actions** taken

## Observations 
* Random motion can teach a lot about getting stuck. Most classic formulations of mapping usually have to hard code getting unstuck... Random motions almost always never get's stuck. Maybe there's something to learn here. 
* I think that we want to try: 
* * Associate perceptions 
* * Sort of predict how perception will look like if we move a certain direction
* Maybe actions can be used to calibrate perceptions. Perhaps not so much in the POMDP sense of if I take this action with this observation, what are the odds of having this other observation. 


## Curiosities

* Can we learn a model of an object, just from viewing it. What kind of data structures do we need to perform this. 
* * I think that we humans encounter the same objects more often than not. Very rarely to we meet absolutely new objects. We meet what we have seen before albeit in changing contexts. 
* If we have sensor readings and actions, can't we have agency over where we want to move in an environment 
* Also how quickly can we explore a new place (with our knowledge of perceptions how actions change perceptions)
* how does robot internalize "you can't go through walls"
* Are there extremely light weight heuristics for going about this problem? 

# Moving using Perceptions - SenseTrack, "SnapBack Perception", The Familiar Trace, EchoTrace
We can work with differences (much like visual neurons, which fire in response to changes in stimulation - https://chatgpt.com/c/68acdbf4-b308-832d-95f4-2f312b81ef30 )

We can create a sort of a graph (whose dimensions is determined by actions available) for the differences.

We can have differences connected by actions. Say that they were obtained during random exploration; the same can be used during active (agentic) exploration to confirm that we are in the same place. This becomes expecially important if we have more than two places that share a part with the same differences. We can use action to confirm where we really are. And perhaps to even add new places (new branches of our graph).

Or we can teach a Convolutional Network to detect features Invariantly from a 1d stream. Then it forms a sort of object detector of some type. This can solve the problem of seeing an object at a distance and at an angle. Also this appears sufficiently novel, but novelty isn't the main point, implementation is.

## 1D Implementation
We have one lidar ray. Also, movement is up and down only. 
We use intensifies, not differences.
We currently detect an intensity... We move randomly (up or down), creating a graph of values, with actions between them.

We might need to determine a quantization (noise level) so that close readings maybe the same.

So say that we dream of moving to a level of intensity of 40, and 40 was near to 35 (which was connected to 25, 20, 15, all via up actions). If we are at 15 currently, we can apply up actions to see if we move to 40. This appears to be a simple exercise, yet this simple mechanism might be thing we exactly need to complete our system.

I think the major issue here is the approcimating two readings as the same.

The graph for this type of system is nodes of scalar value with actions of scalar values as well (integers representing actions)

## 2D Implementation
Sensors - 1d lidar, 1 ray
Actuators - up, down, left
Same as 1d implementation, but add ability to turn left.

A few observations from this is that different places can give the same sequence of readings for up/down movements.

The turn movement is the most different one,... It can give profile of an object, but we aren't speaking form the pov of the robot. We would like to speak from the pov of the robot. 