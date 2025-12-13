## KNN Perception Prediction via Trajectory Matching 

We use trajectory information and history to predict perceptions (with or without actions). 

Let's attempt this
- Say we are at step 0
- We get all sequences that are like step 0 
- We could try find the average of the next steps given action (or without action) ... maybe find % error in this prediction

With windowed, we could try the the following: 
- We are at the first window (step 0 to N-1)
- We take the readings at first step of the window and mark them
- We take the next reading and mark them 

We check which sequences are representative of our current motion (those sequences in the sequences that are near each other and resemble our current sequence)

Clarify if this is a sensible approach