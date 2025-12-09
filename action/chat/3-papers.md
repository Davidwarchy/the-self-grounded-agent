## [Mobile Robot Navigation Based on Deep Reinforcement Learning with 2D-LiDAR Sensor using Stochastic Approach](https://ieeexplore.ieee.org/document/9419565)
@INPROCEEDINGS{9419565,
  author={Beomsoo, Han and Ravankar, Ankit A. and Emaru, Takanori},
  booktitle={2021 IEEE International Conference on Intelligence and Safety for Robotics (ISR)}, 
  title={Mobile Robot Navigation Based on Deep Reinforcement Learning with 2D-LiDAR Sensor using Stochastic Approach}, 
  year={2021},
  volume={},
  number={},
  pages={417-422},
  keywords={Location awareness;Training;Navigation;Stochastic processes;Training data;Reinforcement learning;Robot sensing systems},
  doi={10.1109/ISR50024.2021.9419565}
}

Uses diff drive and 2d lidar to navigate to target

### Critique 
"The relative poses (x, y, θ) between the robot and the goal is calculated using the Hector Mapping [23] at every time-step." basically oracle information

## Substandard papers 
- [Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation](https://arxiv.org/abs/1703.00420) - gets goal position as input 
- [From Perception to Decision: A Data-driven Approach to End-to-end Motion Planning for Autonomous Ground Robots](https://arxiv.org/abs/1609.07910) - get's goal position and worse uses imitation learning 
- [The Impact of LiDAR Configuration on Goal-Based Navigation within a Deep Reinforcement Learning Framework](https://www.mdpi.com/1424-8220/23/24/9732) - uses goal position and direction 

## [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363)
This is close to what I'm talking about 


## Ideas 
### Monte Carlo in Continuous Space 
We want to do something like monte carlo learning, such as was done in the maze... 

There's two issues: 
- distance 
- angle (orientation) 

**Distance**
- The readings (taken as vectors) are highly correlated to the average distance of the lidar rays for each reading 

**Angle** 
From one single point in space, if we move around
- Lidar readings change  
- Average distance changes as well 

**At a single point in space**
- We can have a great deal of readings depending on where we are facing (but if we are facing at given angle, the readings will be the same) 

**Wondering** 

What's the best way for state representation that is robust and strong 

When human beings move about, they have a sense of objects behind them, they have a sense of where things are, even when those things are out of sight. 

There's an issue with rotation: 
- It kinda makes things hard for having unique spatial representations. So a method that is rotation would be helpful. But then such such states (for application in Monte Carlo) would lead us to make guesses about how to move,... would those guesses be better or worse than random walk. 

    
### Arbitrary goal 
I like the idea of using arbitrary goals to learn navigation stuff... this is free learning about spatial dynamics

As of now, the main challenge would be that we are using this method 
- explore about gathering readings 
- cluster the readings as our sense of "objects" 

This approach is weak because it's very dependent on distance and orientation, and embeddings are mixed up at any single point in space. 
### Hindsight Experience Replay (HER)
This might be hinting at something... we are using our experience (even when it doesn't lead to goals) to learn dynamics. 

But how we represent objects is important. If we do it wrong (as with just the raw lidar), we are likely to have even worse than random walk results. 


### 
THere's an assumption by [Gemini](https://gemini.google.com/app/75153070f440d06f) that the agent must:
- Explore the room effectively (without getting stuck).
- Recognize the "geometric signature" of the goal object solely from LiDAR points.
- Stop when the signature matches.

I think that the only relevant one is better exploration... and maybe some recognition (knowing the signature of a desirable object goal). 

[ChatGPT](https://chatgpt.com/c/69316a4f-58f8-832b-802b-7d71c7f59c49)
One single LiDAR scan has this property:
- many very different positions + headings can produce similar scans
- so a single scan is ambiguous 

We can call this the problem of aliasing. 

But as soon as you move:
- the scan from t
- the scan from t+1

together disambiguate.

But we'd need to include action information to fully realize good disambiguation. 

This convo with chat is the closest I came from a reasonable response from an LLM. 

### Scanning Patterns (Learned Scanning Pattern)
- Learned scanning pattern
- I think that people (and other animal) form impression of what things they are touching from a few things
- - scanning (with eyes, or touch). Just a single static touch doesn't reveal things very quick 
- - memory. Using memory of the past to like current observation to 
- - learning. As we are touching and scanning, we are learning about the pattern. if it's new, we can store it in memory 

_C. elegans_ has the way it moves its head