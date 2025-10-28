## Experiment 
let's say I have a 100 ray lidar on a two wheeled differential drive
I do a random walk in a room
at each step I record the step number, action, lidar readings
if I train a contrastive model as follows:
- positive pairs: reading i + consecutive reading
- negative pairs: reading i + random reading 
What am I really teaching the model? 

## [Grok](https://grok.com/c/3a5160e1-312f-4c14-a76f-66b9d54f890c)
* "The model learns to encode features that are consistent across small changes in the robot’s position or orientation." 

What would you say are the features taht are consistence across small changes

* "This creates a structured latent space that reflects the robot’s trajectory and the room’s geometry, implicitly encoding the relationships between the robot’s movements and the environment." 
How is this? 

* "The model isn’t explicitly reconstructing a map of the room or learning a global representation of the environment. Instead, it’s learning local relationships between consecutive scans." 

Write code to get the correlation between the distance to walls and color
distance to walls - maybe average lidar rays (columns are ray_0, ..., ray_99)
color - embeddings -> pca dimensionality reduction to 3d 

Example code is given for access. What is a strong/weak correlation value? 