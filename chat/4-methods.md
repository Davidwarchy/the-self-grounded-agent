- Get embeddings, cluster them. Plot corresponding lidars of each clusters in a diagram. 
- - option to only get lidar readings that are a given temporal distance apart (to show that even temporarily distant places collapse into a similar embedding)
- Get embeddings, cluster them. Plot a cluster on the map 
- Take embeddings, convert to 3d pca, then rgb. Plot in diagram 
- Take embeddings, but just those within a given orientation, convert to 3d pca, then rgb. Plot in diagram 
- Correlations between features (full embeddings or pca-reduced) with human-intuition of distance to wall, openness, turn intensity 


- Option to save as pdf 
- Option to sample the embeddings in case they are many 
- Option to show arrows (including scale of arrow) 

We aimed to test whether we could have something similar to a shape detector, a proxy for objecthood detection. 

We set up an environment with different shapes: triangles, circles, semicircular grooves, etc. 

The hypothesis was that if we trained a model with temporal continuity learning, then we will have a model that is capable of distinguishing the shapes. How this is done is by: 
- training th

## Results 

![alt text](image.png)

It's interesting that a random model shows a clear spatial pattern of clustering even when it's not trained. This is interesting. 

### Neigborhood Preservation 

## Test: Randomly Initialized Network has Structure 
### Setup 
We want to test a theory here: 

That randomly initialized networks will produce structure output for our lidar enabled differential drive even without training. 

Here's what we do. 

- Read lidar readings 
- Pass them through a neural net to give embeddings 
- Convert embeddings to RGB 
- Show RGB embeddings in their locations
- Show RGB embeddings in their locations for a given orientation 

do this in analysis folder

we already have the functions for almost everything 
- gettingn the embedding dimension from metadata 
- getting environment name from meta 
- showing all colored rgb embeddings 
- showing all colored rgb embeddings (oriented)

Make code as simple as possible 

### Results 

![alt text](image-1.png)

It's true: Lidar does have structure even before any training takes place... this is interesting. 

This is an interesting result.