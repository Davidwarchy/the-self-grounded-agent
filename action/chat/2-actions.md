## Reactions to perceptions 
- Do we use the concept of nearness to a given perception or? ie, geometric similarity... we already have millions of readings, with episodes that have rewards 

## Readings and Rewards 
- How do we move from readings to rewards, given the continuous nature of readings

## Previous monte carlo implementation - discrete 
- A reading -> action 
- Each reading -> action 

## Continuous space 
- Here there's a fundamental difference: we have a continuous space 

## What do we do? 
Just some brainstorming 
- Discretize readings, and sort of attach actions to various bins 
- Learn about objects, and react to objects or perceptions close to these objects 
- - Maybe use temporal contrastive learning. How do we know to act when we are at various angle (different from what we trained); how do we choose how long the temporal contextual window should be
- <SOME OTHER METHOD>

## Monte carlo in continous space
- We want to apply the same trick we use in continuous space but in discrete space
- How would we do this (while also being fully embodied - ie, no sketchy tricks that undermine autonomy)

# [Gemini](https://gemini.google.com/app/1e82b5b1a49ebaba)
- Suggestion from gemini
- An interesting suggestion from gemini (that we use clusters as states). This seems like a big time handcrafting, so we want to take note of it if we are going to be using it. 
