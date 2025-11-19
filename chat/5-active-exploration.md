We want to create another part (in another folder seperate from exploration and training) that does active moving to certain objects... it does this even when it got lost during exploration. So we have a chicken and egg problem (of knowing the objects in order to move to them; and moving about in order to discover object hood!)....

We want to have the robot be biased to moving towards just a few objects (it's interesting how lifeform ever pay attention to just a few stimuli and not the whole barrage of stimulation coming in). 

I don't want us using heavy classical navigation techniques. 

The basic idea is that we can use our knowledge of objecthood, actions (random walk, uniform distribution of random walk sequences, levy, etc) to move towards objects. Perhaps train a neural net to do this. 

The assumption is that the system will capable of the factors that might be helpful in object goal navigation, including (probably): 
- long term association of movements 
- contextual information of where it is

I say probably coz we humans aren't very good at expressing the factors that an intelligence would consider to make it successful. 


What's a good balance between exploration and exploitation... Is there a way to encode this in a natural way that is free from human handcrafting. 