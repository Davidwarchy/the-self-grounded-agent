## [Extending sensorimotor contingency theory: prediction, planning, and action generation](https://www.researchgate.net/publication/258610097_Extending_sensorimotor_contingency_theory_Prediction_planning_and_action_generation?enrichId=rgreq-b77a4c35de9398d5e02df29a9c64d541-XXX&enrichSource=Y292ZXJQYWdlOzI1ODYxMDA5NztBUzoxMDI0NzcxMzUzNTE4MjFAMTQwMTQ0Mzg2MjAyMw%3D%3D&el=1_x_3&_esc=publicationCoverPdf)
This paper, by Alexander Maye and Andreas K. Engel, builds on sensorimotor contingency theory (SMCT), which says perception comes from understanding how actions change sensory inputs (like moving your head changes what you see). The authors extend this idea to show how robots can use these "sensorimotor contingencies" (SMCs) for predicting, planning, and choosing actions. Here’s a simple summary:

### Main Ideas:
- **Core Concept**: SMCT suggests perception isn’t about building a mental picture of the world but about learning how actions affect sensory inputs. For example, turning a robot’s wheels changes what its sensors detect.
- **Extended SMCs (eSMCs)**: The authors use a model where robots store sequences of actions and sensory outcomes (eSMCs) to predict what happens next and plan actions. These sequences are saved in a tree-like structure, with each action linked to a "utility" score (how useful or costly it is, e.g., avoiding collisions saves energy).
- **Action Selection**: The robot picks actions that maximize utility, balancing goals like avoiding crashes and saving power. It does this by simulating possible action outcomes using its stored eSMCs.
- **Robot Experiment**: They tested a robot in a simple rectangular space, where it had to move without hitting walls. The robot learned to avoid collisions by making smooth U-turns, which was unexpected and showed it could find smart solutions on its own.
- **Resilience**: Even when its distance sensors were turned off, the robot could still avoid some collisions using past eSMC knowledge, showing a basic form of "dead reckoning" (estimating position without sensors).

### Key Points:
- **Prediction and Planning**: The robot uses eSMCs to predict future sensory inputs and plan actions, like a human anticipating where a ball will land.
- **No Internal World Model**: Unlike traditional robot designs that build a detailed map of the environment, this approach uses the environment itself as the "model," sampled through actions.
- **Awareness**: The robot’s ability to predict "what if" scenarios for different actions suggests a basic form of awareness, as it can plan rational behaviors.
- **Challenges**: The approach works well in simple settings but may struggle with complex environments (e.g., high-resolution sensors like cameras) due to memory and processing limits.

### Why It Matters:
The paper shows how SMCT can guide robot control, making them more adaptive and efficient without needing complex internal models. It also suggests robots could mimic biological learning, like how humans or animals act and perceive. The authors hope this inspires better robot designs and deeper studies into how perception and action connect.

In short, it’s about teaching robots to learn, predict, and act like living creatures by understanding how their actions shape what they sense, leading to smarter, goal-driven behavior

### Thoughts (some of them are scattered, and might not be suitable for implementation in one setup... they are just thoughts coming to mind)

[Grok](https://grok.com/chat/677a042b-b951-408f-a5ce-e26c2edcdce8)
"Then answers to questions such as what structures should be considered in the changes of sensory signals and how to extract them, how to explore huge or even infinite action spaces, how to memorize knowledge of SMCs, and many others have to be found." 

**Sensorimotor Contigencies (SMCs)**: Perception emerges from an agent's active exploration of the environment through actions, which produce predictable patterns of sensory changes called Sensorimotor Contingencies (SMCs)

If we applied a convolutional filter to perceptions, it's capable of detecting invariances with motion, but seems to introduce (very subtly) the assumption of continuity of objects or space. It's good to take note of this strong assumption. 

"The robot's "purpose" (or goal) is to navigate to a charging station in the room while avoiding collisions and minimizing energy use. We'll contrast two scenarios: one where action selection is random (not tuned to relevant SMCs), and one where it's purposeful (tuned to goal-relevant SMCs). This shows how action selection "tunes" the robot to perceive its environment effectively." ~ [Grok](https://grok.com/chat/677a042b-b951-408f-a5ce-e26c2edcdce8) - this is exactly the kind of stuff I think is important... 

We are trying to find a **tuned action strategy** to get to goals. 

We may even get to a point of associating "multiple actions" to open lidars (ie, when the way forward is open, we very much likely to move forward a lot). I'm thinking that right now, that caribou (reindeer) kids will follow their moms soon after birth... I think that some sort of action patterns are already installed in the caribou calf before birth (and this together with the instinct to follow the mother). 

so sensorimotor contigencies are predictable patterns of perception that come from taking action? 

if the utility score is incrase open-space readings (this seems to be a handcrafted, local minima thing... given goal of finding a charging station which might be at a closed place, at a corner or wall probably). 

I think that what we want is a couple of utility functions (or the ability to have multiple utility functions - whose purpose is two fold : increase odds of getting out of a dysfunctional utility, also diversifying utilities... this seems to be an idea that can be explored later... but remember to keep language simple and grounded in physical objects and experiments)

We could have a high level neuron that fires when we do a sequence of actions. A sequence of actions might be suitable for something like approaching a desirable stimuli (say a charging station when power is low). We can have one neuron that does this. This sounds similar to how higher level neurons in the visual system get triggers by certain patterns. Here high level neurons trigger certain motor motors (the exact reverse of vision). 

I'm wondering if something like context and perspective can arise from the perceptions/actions. 
* Context here means that we are kind of aware of of things around us (which we can confirm with actions towards those things) 
* Perspective means that we can identify a range of objects (things) from a vantage, and know that those are specific things (maybe associated with given actions). For example if we saw two profiles, one for something desirable (with an approach bias) and another undesirable (with an avoid bias), we might want to see the systems for approach/avoid being fired strongly. 

We can associate long sequences of actions with perceptions or vice versa. This can be flexible and a bit arbitrary, I'm thinking. Like we could have. 

Action is part of perception. Ie, we can reduce uncertainty about what we are seeing by performing action. A sequence of perceptions can confirm an object's or a known profile. 

"A separation of actions with respect to these two functions seems conceptually awkward; therefore, we aspire to develop a model of action generation that considers perception and goal achievement in an integrated manner." 

"Our conception of the term ‘‘action’’ therefore includes the notion of goal-directedness (McGann, 2007)." 

* I've always thought this to be true. But it's perhaps time I concretized it and implemented systems with it. 

The idea that we can use CNNs to identify objects (even if they changed context). This is really impressing on me right now. 

Sutton says that **search** and **learning** are the most scalable techniques. **_Search_ and _learning_ are highly scalable for building intelligenc**e — for solving complex problems, making decisions, and achieving goals in environments that are too complicated for human-crafted rules.

Fire together wire together mechanisms naturally incorporate uncertainty. For example if we sort of perform two separate actions for a given stimuli, we get strong connections to both actions. 

"It suggests instead that the brain’s sensory processing serves the preparation of several potential actions in parallel that are afforded by the current context. These potential actions compete against each other. The competition is biased by accumulating sensory information about the aptness of different actions and by top-down information. This framework is supported by an elegant reinterpretation of neurophysiological data on action-related brain activity (Cisek & Kalaska, 2010)" 
In my subjective experience, I think that the actions taken by the brain are very highly conditioned by recent information affecting objects in the prevailing cnotext. For example, if I hear some damning stereotype against muslims, it will pop up to mind when I encounter a muslim next time. This conditioning can occur even within the brain itself (like during reflection about something; which usually ends up with a person changing their attitude or reactions). 

The idea that we sentient beings perceive the environment without creating 3d models of the environment is ancient (1979) by [The ecological approach to visual perception - Gibson](https://berlinarchaeology.wordpress.com/wp-content/uploads/2022/03/gibson-james-j.-19862015-the-ecological-approach-to-visual-perception-2015-re-issue.pdf)

I think there's something to say about reinforcemnt learning, whether or not our scheme succeeds. 

I might want to reaad [Multilevel structure in behaviour and in the brain: a model of Fuster’s hierarchy](https://royalsocietypublishing.org/doi/pdf/10.1098/rstb.2007.2056) but without a need to replicate what the brain does. Just a functional system. It appears to be an interesting paper. 

Also this [A discrete computational model of sensorimotor contingencies for object perception and control of behavior](https://ieeexplore.ieee.org/abstract/document/5979919)

I think that we can have an implicit utility for speed, which is sort of natural. 

How would averaging affect perception? Let's say that we averaged incoming perceptions (and maybe even more interestingly actions). I'm wondering whether it would work out into something good (maybe help with extrapolation). 

I think that the good thing with this (if we implemented it) is that we need not do handcrafting for something like: 
* buffer between walls to accomodate robot size (the robot sort of implicitly learns to avoid walls) 
* 
How does it learn to get unstuck??? 

Perception, control (action), learning can be combined in one step. Ie, when we are confirming

### Critique
* Very simple environment (a rectangular room). 
* Though the goals are useful (roaming while minimizing collisions, acceleration and energy consumption), it seems devoid of any utility to autonomous agents themselves or to human beings. 
* "In our approach jerk is a component of the utility function which governs learning in the robot." so much handcrafting 
* Very abstract language (without concrete examples), something that limits understanding... 
* 

