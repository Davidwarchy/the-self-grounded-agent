We want to create another part (in another folder seperate from exploration and training) that does active moving to certain objects... it does this even when it got lost during exploration. So we have a chicken and egg problem (of knowing the objects in order to move to them; and moving about in order to discover objecthood!)....

We want to have the robot be biased to moving towards just a few objects (it's interesting how lifeform ever pay attention to just a few stimuli and not the whole barrage of stimulation coming in). 

I don't want us using heavy classical navigation techniques. 

The basic idea is that we can use our knowledge of objecthood, actions (random walk, uniform distribution of random walk sequences, levy, etc) to move towards objects. Perhaps train a neural net to do this. 

The assumption is that the system will capable of the factors that might be helpful in object goal navigation, including (probably): 
- long term association of movements 
- contextual information of where it is

I say probably coz we humans aren't very good at expressing the factors that an intelligence would consider to make it successful. 


What's a good balance between exploration and exploitation... Is there a way to encode this in a natural way that is free from human handcrafting. 

{202511191511:
# Experiment
We want to perform this experiment 
Associative memory 
Transformer (maybe or not)
Uniformly distributed run lengths 
Attaching certain actions (runs) to certain perceptions
---

The agent moves about (taking in lidar readings)

As it moves, it creates embeddings through its perceptual network (the contrastive learning one) 

Also as it moves, it also **stores percepts and actions into an associative memory**

Also as it moves, it **retrieves percepts and actions from the associative memory**

We want to see how good it is at returning to certain percepts, even after long explorations.

There's a bootstrapping problem here: 
- the percepts develop from moving about 
- 

---
Are neural networks the thing to help with this? I think that they are not. But if not, what else might be a good alternative? 

When we think about how we move about: we derive a lot of informational cues from the environment. How would we have a system that learns to extract informational cues (eg, by scanning around the room). We don't want to handcraft this sort of thing. We want to have a system that is capable of exploring this space of intelligence. It's not clear how neural networks can do this? What makes language really convenient for transformers and learning is that we have a lot of examples of it. It's already perfect data out there. There's not so much for robots (perhaps there are datasets that one might want to look into). And I think that if we had a way of perfecting trajectories from random motion, then we would have something even more beautiful than the massive datasets of language. That is: language is limited by human beings - there's only so much "pure data" as there are humans (even if LLMs have become really good at writing and producing language). If we can have a way of getting *fit* paths from random explorations, then we have something massive than language datasets. We will then essentially be constrained only by the number of active agents exploring the world, not by human beings producing trajectories. 

"LIFE — Latent Information for Fitness Enhancement" sounds like a good name for a system. 

We can have a system that does random walks be sufficient to survive. Again we might want to give our agents abilities, rather than optimizing for shortest paths to objects or something like that. 


} 

{202511221223:
# More ineresting experiments 
I recently tried experiments on inverse dynamics (trying to predict actions from lidar readings) and it gave impressive results (97% accuracy). I want us to try a couple of more interesting experiments. 

I think that most interesting would be going to a place with a particular shape 

How about we try encoding the readings and actions into something that is really invariant. 

What about we took a couple of past n readings and we try to encode this instead of just using a single reading. What if we used past n readings... how would the architecture inputs, outputs, look like?

## Forward Dyanmics

I did try to implement this, and it's not producing good results. 

![alt text](image-2.png)

## Autoencoder 

Let's try to see if we can reconstruct lidar readings. 

The setup is simple: 
- Inputs lidar reading
- Encode to latent dimension 
- Output reconstruction 

}

{202511231041:


# Goal Encoding, Goal Navigation 
Where I am currently - the types of objects and a sense of their layout 
How to get to some other place 
Encoding goals and moving towards goal (ie, how do we encode goals as a pattern of perceptions and actions) 
How do we tie this into something that we can generalize to other environments 

Currently our system is able to highly correlate with distance from obstacles... this is an interesting results, but I think that we can do better. I think that we can get a system that encodes a whole lot of place

# References 
## [LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures](https://arxiv.org/pdf/2312.04000)
The paper addresses a critical bottleneck in Self-Supervised Learning (SSL): **how to evaluate the quality of learned representations without running expensive downstream tasks**.

I think that the answer to this is that, we don't know of how to evaluate this... (at least in the context of alife). 

### Critique 
tbh, this appears to be very sketchy
it's right at the heart of what I'm studying right now - developing perception for fitness
the problem with the paper is that it ultimately makes the mistake of using a human designed mathematical notion as the thing we are optimizing, so we can't really apply it to our life on silicon paradigm. It's not related to fitness. 
We just want to to develop perception that helps us survive, the problem is that survival doesn't have a quantifiable mathemtical loss function (unless we assume some form of central control). 
I would love to hear more about this sort of things from other people, papers, videos

## [A Survey on Self-Supervised Representation Learning](https://arxiv.org/pdf/2308.11455)

Says that images often contain information that is irrelevant for downstream tasks. 

Good representations are those that have good downstream properties. 

## [Fully autonomous robots are much closer than you think – Sergey Levine](https://www.youtube.com/watch?v=48pxVdmkMIE)

Continuous action model; 

Using only high quality is bad! The robot makes mistakes. Humans rarely make mistakes. Robots need to know how recover from mistakes, so mistakes are essential. All data collected by robot operators. 

Realistic. Job. Replace paper towels. Fold laundry. Assemble cups or boxes. 
} 

{202511250606:
## [Evolution of Rewards for Food and Motor Action by Simulating Birth and Death](https://direct.mit.edu/isal/proceedings-pdf/isal2024/36/35/2461175/isal_a_00753.pdf)
[Code](https://github.com/oist/emevo)

@inproceedings{kanagawa2024evolution,
  title        = {Evolution of Rewards for Food and Motor Action by Simulating Birth and Death},
  author       = {Kanagawa, Yuji and Doya, Kenji},
  booktitle    = {Artificial Life Conference Proceedings 36},
  year         = {2024},
  publisher    = {MIT Press},
  url          = {https://mitpress.mit.edu/},
  note         = {ALIFE 2024}
}

Evolution of the reward system. 

### Robot 
**Sensors**: 16 ray lidar (implemented as distance sensors); A tactile sensor detecting physical contact; Proprioception: Velocity, Angle, and Internal Energy level
**Actuators**: 2 wheel differential drive

### Critiques
Too much assumption for reality: reproduction (how would this happen in real metal/silicon)

"positive rewards for food acquisition and negative rewards for motor action can evolve from randomly initialized ones" This is interesting - the claim is that fitness rewards can emerge rom random rewards. 

I'm not surprised that there's positive rewards for motor actions, since we tend to reward actions that lead to bigger rewards. There's something good and bad in my argument. 

How are the sensors of the robot? Actuators? 

It's decentralized

Simpified model of death. Death can be due to a great deal of factors, eg, physical damage, dysfunction (disease)

A lot of copying from natural life. Asexual reproduction is assumed (how would it be realized in practice?), need for hunting food (we don't know about this in the real world)... 

I'd imagene. 

I think that there was already assumed that food and energy are good. But in reality evolution never really knows if something is good. 

Eating food 

#### On Emergence of Fitness-relevant Rewards
"positive rewards for food acquisition and negative rewards for motor action can evolve from randomly initialized ones" I now find their claim to be false. It appeared to be something like "We have a robot with lidar rays and differential drive. We make it move about the environment optimizing for random things, but it ends up optimizing for food and motion." But it seems that they just optmized for food and motion, and got what we expected. 

I was a bit wrong and too harsh here. 

The reward function is like: $$r = w_{food} \cdot \text{food\_intake} + w_{act} \cdot \text{action}$$

They then randomize the weights. They initialized the agents with random weights ($w_{food}$ and $w_{act}$)

- Some agents started out hating food (negative weight). They avoided food, starved, and died.
- Some agents started out hating movement. They sat still, starved, and died.
- The "evolution" was the survival filter. Only the random agents that happened to have $w_{food} > 0$ survived long enough to reproduce. 

While I was harsh, I had something correct: The researchers did pre-determine the inputs to the reward function. They hardcoded the reward equation to look like this:

$$r = w_{food} \cdot \text{food\_intake} + w_{act} \cdot \text{action}$$

So what the experiment was essentially doing was emerging the +/- sign on he weights. We are essentially evolving a reward function constrained to food intake and action. 

The critique that **"evolution never really knows if something is good"** is the most poignant. In this simulation, evolution did act as the judge, but the researchers rigged the jury by only letting the reward function "see" food and motion. A truly open-ended evolution would have let the agent choose its own reward inputs from raw pixel/lidar data, rather than pre-processed "food eaten" counters.

"Agents that accidentally evolve a network that outputs Positive Reward when Lidar detects small circular objects (food) will learn to approach them via Reinforcement Learning." ~ [Gem](https://gemini.google.com/app/a6b9d06458f97e82)

Not really. 

I think that the fundamental thing will be something like this: 

Agent rewards internal state like high energy. RL does reinforcement for us, making us reinforce movements that lead to the high energy. This is what will happen in fit agents. This might backpropagate into something like developing fast movements towards food-like blobs. 

In agents that aren't fit, the opposite happens. It makes the wrong guesses, rewards bad actions, and punished by death. 

## [Evolution of Fear and Social Rewards in Prey-Predator Relationship](https://arxiv.org/pdf/2507.09992)

@article{kanagawa2025evolution,
  title        = {Evolution of Fear and Social Rewards in Prey-Predator Relationship},
  author       = {Kanagawa, Yuji and Doya, Kenji},
  journal      = {arXiv preprint arXiv:2507.09992},
  year         = {2025},
  url          = {https://arxiv.org/abs/2507.09992}
}

This is by the same author as [Evolution of Rewards for Food and Motor Action by Simulating Birth and Death](https://direct.mit.edu/isal/proceedings-pdf/isal2024/36/35/2461175/isal_a_00753.pdf). They make the same ommision as in the other paper, of specifying the reward function in terms of human known fitness proxies. 

It's interesting that reward functions don't appear in classical life until we have high level animals. Bacteria and archaea basically have come to exploit luck and clever strategies (levy walks) in order to get to food. At what stage do reward functions appear in life? At what level of complexity? 

https://gemini.google.com/app/7903122717f63ddf
}

"The core idea is that dopamine doesn't just signal "good," it directly instructs the nervous system to strengthen the specific neural connections that were active just before the reward arrived." ~ [Deep](https://chat.deepseek.com/a/chat/s/f83e9607-62d3-422e-8e4e-006a26f9da4e)

## C elegans

[Gem](https://gemini.google.com/app/da73736d5408d9e6) gives a good description of the mechanics of _C. elegans_

### [THE GENETICS OF CAENORHABDZTZS ELEGANS](https://wormatlas.org/papers/Brenner1974.pdf)
@article{brenner1974genetics,
  title={The genetics of *Caenorhabditis elegans*},
  author={Brenner, Sydney},
  journal={Genetics},
  volume={77},
  number={1},
  pages={71--94},
  year={1974},
  doi={10.1093/genetics/77.1.71}
}
[c elegans video](https://www.youtube.com/watch?v=zjqLwPgLnV0) nice video on the nematode 

- genes specify the structure and function of the nervous system

![alt text](image-3.png)

### [The Fundamental Role of Pirouettes in Caenorhabditis elegans Chemotaxis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6782915/)
@article{shimomura1999pirouettes,
  author    = {Pierce-Shimomura, J. T. and Morse, T. M. and Lockery, S. R.},
  title     = {The fundamental role of pirouettes in \textit{Caenorhabditis elegans} chemotaxis},
  journal   = {Journal of Neuroscience},
  year      = {1999},
  volume    = {19},
  number    = {21},
  pages     = {9557--9569},
  doi       = {10.1523/JNEUROSCI.19-21-09557.1999}
}

The abstract explain things well: 

"""
To investigate the behavioral mechanism of chemotaxis inCaenorhabditis elegans, we recorded the instantaneous position, speed, and turning rate of single worms as a function of time during chemotaxis in gradients of the attractants ammonium chloride or biotin. Analysis of turning rate showed that each worm track could be divided into periods of smooth swimming (runs) and periods of frequent turning (pirouettes). The initiation of pirouettes was correlated with the rate of change of concentration (dC/dt) but not with absolute concentration. Pirouettes were most likely to occur when a worm was heading down the gradient (dC/dt < 0) and least likely to occur when a worm was heading up the gradient (dC/dt > 0). Further analysis revealed that the average direction of movement after a pirouette was up the gradient. These observations suggest that chemotaxis is produced by a series of pirouettes that reorient the animal to the gradient. We tested this idea by imposing the correlation between pirouettes and dC/dt on a stochastic point model of worm motion. The model exhibited chemotaxis behavior in a radial gradient and also in a novel planar gradient. Thus, the pirouette model ofC. elegans chemotaxis is sufficient and general.
"""

It's exactly like run and tumbles in _E. coli_. 


These might show my journey to understanding motion in _C. elegans_: 
- https://chatgpt.com/c/69267b23-7688-8327-8596-ad1de7c27ac0
- https://gemini.google.com/app/da73736d5408d9e6

### [C. elegans II](https://www.ncbi.nlm.nih.gov/books/NBK20005/)
@incollection{riddle1997mechanosensory,
  title = {Mechanosensory Control of Locomotion},
  author = {Riddle, David L. and Blumenthal, Thomas and Meyer, Barbara J. and others},
  booktitle = {C. elegans II},
  edition = {2nd},
  editor = {Riddle, David L. and Blumenthal, Thomas and Meyer, Barbara J. and Priess, Judith R.},
  publisher = {Cold Spring Harbor Laboratory Press},
  address = {Cold Spring Harbor, NY},
  year = {1997},
  pages = {Chapter II, (exact pages if known)},
}


book says that the stimuli that celegans reacts to are: 
- mechanical (touch) 
- chemical 
- thermal 

structure of c elegans 
- a tube 
- sensors 
- - temperature (thermal)
- - chemical (chemical)
- - touch (mechanical) 
- - no eyes, or long range sensors 
- muscle cells
- - mouth cells (20 cells) 
- - intestinal cells (2 cells)
- - defecation cells (2 cells) 

- - Sex-specific muscles
- - - Hermaphrodites add: 8 vulval, 8 uterine, Contractile gonadal sheath
- - - Males add: 41 mating muscles

c elegans has 959 somatic cells (everything except sperm/eggs): 
- 302 neurons 
- 95 body-wall muscle cells
- 20 pharyngeal (mouth) muscle cells


"B-type motor neurons” aren’t just ordinary neurons — they integrate proprioceptive feedback and drive rhythmic locomotion"
I was stuck on this point for a long time. Bascially from morning to afternoon. 

### [Proprioceptive coupling within motor neurons drives C. elegans forward locomotion](https://pmc.ncbi.nlm.nih.gov/articles/PMC3508473/)
@article{wen2012proprioceptive,
  title={Proprioceptive coupling within motor neurons drives C. elegans forward locomotion},
  author={Wen, Qing and Po, Matthew D and Hulme, Elizabeth and Chen, Sherry and Liu, Xinyu and Kwok, Simon W and Gershow, Marc and Leifer, Andrew M and Butler, Victoria and Fang-Yen, Christopher and Kawano, Tomohiro and Schafer, William R and Whitesides, George and Wyart, Matthieu and Chklovskii, Dmitri B and Zhen, Mei and Samuel, Aravinthan D T},
  journal={Neuron},
  volume={76},
  number={4},
  pages={750--761},
  year={2012},
  publisher={Elsevier},
  doi={10.1016/j.neuron.2012.08.039},
  url={https://doi.org/10.1016/j.neuron.2012.08.039}
}
"Here, we examined whether the worm motor circuit has proprioceptive properties and how these properties are connected to undulatory dynamics." One would better understand this if they have a good picure of SENSORY NEURONS -> INTERNEURONS -> MOTOR NEURONS -> MUSCLE CELLS

I found this to have a good intro on c elegans

"How the C. elegans motor circuit organizes bending waves along its body during locomotion is poorly understood." So this was true circa 2012

Let's define a segment as a cylindrical section, part of the larger cylinder. 

"How the C. elegans motor circuit organizes bending waves along its body during locomotion is poorly understood. Even when all premotor interneurons are ablated (Kawano et al., 2011; Zheng et al., 1999), C. elegans retains the ability to generate local body bending, suggesting that the motor circuit itself (A, B, and D-type neurons and muscle cells) can generate undulatory waves. However, the synaptic connectivity of the motor circuit does not contain motifs that might be easily interpreted as local CPG elements that could spontaneously generate oscillatory activity, e.g., oscillators driven by mutual inhibition between two neuronal classes that can be found in larger animals (Figure 1B). The synaptic connectivity does contain a pattern to avoid simultaneous contraction of both ventral and dorsal muscles; the VB and DB motor neurons that activate the ventral and dorsal muscles also activate the opposing inhibitory GABAergic motor neurons (DD and VD, respectively). However, this contralateral inhibition generated by GABAergic neurons is not essential for rhythmic activity along the body or the propagation of undulatory waves during forward locomotion (McIntire et al., 1993)." 

this is rich. 

Totally inspired by the paper to do something like a simulation of a worm in webots
- [Design and Evaluation of a Snake Robot withBio-inspired Locomotion over Uneven Terrain](https://www.ijmerr.com/2025/IJMERR-V14N1-78.pdf)

"C. elegans moves forward on its side by propagating dorsal-ventral body bending waves from head to tail." Also this paper confirmed that c elegans moves forward on its side. 

I think that also relevant to the concept of self-grounding is how control system exploit the mechanics (motor neurons propagating undulation)

### [Inhibition Underlies Fast Undulatory Locomotion in Caenorhabditis elegans](https://pmc.ncbi.nlm.nih.gov/articles/PMC7986531/)
@article{deng2021inhibition,
  author    = {Deng, Lin and Denham, John E. and Arya, Chandni and Yuval, Oded and Cohen, Nir and Haspel, Gal},
  title     = {Inhibition Underlies Fast Undulatory Locomotion in *Caenorhabditis elegans*},
  journal   = {eNeuro},
  volume    = {8},
  number    = {2},
  pages     = {ENEURO.0241-20.2020},
  year      = {2021},
  doi       = {10.1523/ENEURO.0241-20.2020}
}

pretty interesting that inhibitions makes things fast in _C. elegans_ 

