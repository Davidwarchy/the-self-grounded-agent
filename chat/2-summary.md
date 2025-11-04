# Repo Organization 
I want us to organize our repo, so that it's easier to analyse and work with it

## Data Collection 
**Pure random walk**: Each step is independent and of equal length.
**Lévy walk**: Step lengths (or durations) follow a heavy-tailed (often power-law) distribution — meaning you get many short moves and a few long ones, creating bursts of exploration.
**Lévy-like random walk**: If the step-length distribution is uniform, it’s sometimes called a variable-step random walk or random walk with variable step lengths.

## Temporal Continuity Learning (TCL)
- Contrastive learning 
- BYOL 
- Other Methods 

## Goal Navigation 

### Spatial ACtive Dynamics 
- If I saw something a few moments ago and made these movements, it's likely I will see it again if I make these other movements 
- If I saw something before passing through a door, I'll likely see if I repass through the door again 
- If I saw something nad moved right, I'll see it again if I move left. Same as backward and forward 
- Encountering walls doesn't lead to a lot of action or progress towards goals 

**Notes**
- These are examples of dynamics an agent might experience as it navigates through the world. Some other dynamics might be difficult to articulate 
- From the dynamics learned, we get to know the choiciest samples of actions/durations for a desired goal (assuming a levy-like random walk) 

**Question**
What is a good paradigm (setup) that allows the learning of such dynamics in an open-ended way

### Navigatioin and Generalization 

- Learn representation (with TCL)
- Learn navigation (How???)

How??? I think that we could have a sequential memory of 
- representations (percepts) 
- actions 
- goals 

How would we generalize what we have stored in the sequential memory (and have it deal with noise and shifts - ie, if we saw a door from a different perspective, how do we still go through it, even if we are a new pose?). Ie, how does learning spatial dyanmics in one room become useful knowledge for navigating in another room? 

We could have a neural net that takes in memory of our recent experiences (perceptions, actions). The idea I'm having about selecting unique recent perceptions is still handcrafting, and we saw that this doesn't generalize well. Can we have a neural net (NN) do all the work on its own? 

Also, if our approach doesn't work, we could still use the Levy walk or random walk to get to where we want. AFter all, this might be the important thing. 