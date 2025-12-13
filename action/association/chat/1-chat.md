Now let's do something here: 
- We will discretize and downsample the readings (downsample to 3 pixels; discretize to 3 levels for each pixel)... Do averaging to downsample (not picking every, for example 33rd ray if we have 100 rays total)
- let's still do random walk 
- We'll use environment 12.png 
- Place reward at 15, 5 (with a radius of 5) 

save data to a folder in the main folder output/downsampled/ (create one if not done yet). Maintain timestamp saving. 

Do random walk and just write the outputs. Indicate when we touch the target with reward variable. 

We will store only k=3 rays (ray_0, ray_1, ray_3) or the number of pixels we downsample to. 

We will have 5 actions 
- up 
- down
- left 
- right 
- nothing 

no run lengths yet 

Write new scripts for this 

Maintain differential control (nothing much is changing except: downsampling/discretization of the rays; saving folder, etc). The input space is discrete, but not the output. Rewrite all files anew, such that we can run them clean