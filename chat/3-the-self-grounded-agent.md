## Life & Fitness 
Definitional problems arise when we try to define life in a concrete way. These concrete definitional difficulties can be attributed to a property of life itself, and rather than seeing it as a problem, it can be viewed a property of the system. Open-ended life can’t be defined because it keeps changing its own rules. The same force that makes life endlessly creative also makes it impossible to pin down. Its resistance to definition isn’t a mistake — it’s what life is. Nonetheless, higher order properties can be used to define requirements and hallmarks of life as in Kriebisch2025: 
- Self-sustaining (though there are obvious exceptions to this)
- Self-replicating
- Randomly Mutating 
- Open-endedly expansion 

Although not explicitly a definition, Schrodinger1944, mentions: 
A form of orderly and lawful behavior of matter that actively evades the decay to thermodynamical equilibrium (death). Organisms keep alive by continually drawing "negative entropy" (order) from their environment. 

What is fitness wrt to this - isn't it the capacity of an organism to survive and do what? 

It is **not**:
- Strength
- Speed
- Intelligence
- Happiness
- Complexity

It **is**:
- **How many surviving offspring (or copies) per unit time, on average, under real selection pressure.**

Life **Fisher1930** does not provide a single formal **definition of life**, but implicitly frames it through the lens of reproduction and population dynamics. An organism is treated as an entity subject to death, reproduction, inheritance.  **Fitness** is given a precise, quantitative, and objective definition: rate of increase of that genotype's lineage. 


## Life as is, Life as it could be 
We do a comparison between life as we generally know it, and life that could be afforded by computer technology. 

Fitness is not strength, speed, or efficiency. 

### Life as is
- Majorly dominated by carbon-based lifeforms. 
- Information is stored in DNA. 
- Expresses a high degree of open-endedness
- Natural selection is the major shaper, alongside mutation, reproduction and struggle for survival. Varition, selection, replication. 

### Life as it could be 
Life on silicon
Presents a most exciting objective 
Robotics is currently well developed that it provides a good medium for things that would be required for fitness: 
- perception (lidar, acoustics, cameras, infrared, radar, )
- motions (moving - motors, wheels, tracks, legs, propellers, thrust engines, wings, etc)
- manipulation (dealing with obstacles, making tools, creating fitness relevant objects, and maybe even replication - von neumann). As discussed in the next section, we see that motions and manipulations have no apriori knowledge of their fitness value. They are often guessed in classical life, and this provides with a platform for bootstrapping our evolution on machines. 
- sequencing of manipulations into fitness-useful activities. 

Also silicon (semiconductor computer based systems) provide a uniform substrate of perception for all modalities. We only need to have the right transducers from the modality of interest into bits. This is one major advantage of semiconductor machines. Visual, tactile, auditory, temperature, etc, can also be converted into a common thing.

Divorce of control and morphology. 

Sometimes learning is embedded in morphology, some morphologies make learning easier. 

Neural nets have been proposed as suitable substrates for the control systems (Doncieux2015EvolutionaryRobotics), subjecting their parameters to mutation and selection. 

On top of that, information replication is superfast in semiconductor substrates. 

## Right Actions on Right Objects 
- A lot fitness activities revolve around approaching nutrients, avoiding repellents or harmful stimuli
- Other parts involving fitness involve having internal organization that sustains the mechanism for approaching/avoiding stimuli alongside other activities. 
- In higher level animals (higher level?) like human beings, critical activities are usually limited to a few critical ones: 
- Evolution has no apriori knowledge of right actions or right objects (stimuli). 
-- In small creatures, the molecular structure itself responds to stimulus to approach/avoid attractants/repellents 

In order for this to work on our life on silicon platform, an agent needs to have a good system for learning object hood. We propose temporal continuity as something that can do this pretty well, and present experiments to highlight this. 

## Goals, Temporal Continuity, Objecthood 
When an agent moves through the world, the continuity of sensory flow naturally induces invariants—patterns that persist despite motion. These invariants become proto-objects. Temporal contrastive learning operationalizes this process: by aligning embeddings of temporally adjacent frames, the system extracts whatever features remain stable through motion. The resulting space encodes not symbols, but sensorimotor invariants—the perceptual atoms from which “things” emerge.

Sometimes the object relevant to fitness isn't within line of site view of the creature. How do simple celled organisms resolve this? 

Does an intelligent system need to make assumptions about continuity, identity, eg, that objects tend to move continuously in space, that objects are embodied in a single body and don't just spontaneously appear and disappear. Would such an intelligence work where continuity and identity weren't a thing. What kind of intelligence would this be? 

Questions we have could be: what would be a framework that easily expresses richness similar to that which we see in naura: 
- agents following moving objects (moving goals)
- scanning behaviour while pursuing goals 
- preferences for moving along walls, etc... 

Also 
**Line of Sight/Reach**
- Small creatures (like bacteria and archaea) have a chemotaxis system that responds directly to stimuli temporal gradients (too small to detect spatial gradients). Explorations are enabled by runs and tumbles. They use exploration and biased random walk. 
- How do large creatures react to stimulus within line of sight. Imagine baby animals that react to images of their mothers with movements towards them. Or some multicellular organism. Let's expound on this
**Out of Line of Sight/Reach**
- Small creatures might not have something that reacts to this. It's amazing that they can survive solely on this. Currently out of sight objects/stimuli can only be reached by exploratory runs and tumbles (these happen even in homogeneous media). 
- Larger creatures: how do they do this? 
- - Many animals (e.g., sharks, bears, dogs) have an extremely acute sense of smell, allowing them to detect concentrations of airborne or waterborne molecules from great distances. Like chemotaxis, this often involves following a concentration plume or gradient to locate the source, which is far out of visual sight.

## Perceptual Darwinism 
In literature and representation learning, perception is divorced from fitness. 

Evolution isn’t guided. Mutations happen randomly; natural selection decides what survives. If something seems unexplained, study selection, not magic causes. ~ Fisher1930 This is The Fundamental Theorem of Natural Selection by Fisher. 

There's no aprior way of knowing what actions to commit on certain perceptions. We guess, and then natural selection does the job. 

Our proposed system provides a mechanism that is easily fitness actionable (ie, fitness ready). 

## Sources of Knowledge 
Knowledge here refers to the performance of right actions on the right objects. Braitenberg pointed out
- Inbuilt - knowledge is inbuilt (knowledge of attractants/repellents or of more complex actions) in simple creatures 
- Cognitive biases - creatures with the right biases survive, while those that get it wrong are culled out by natural selection. Some of the biases can be developed during the lifeform's lifetime, and while being ineffient, need not be fatal. Nonetheless, the lack of aprior knowledge of fitness value is the same. 
- Teachers - experienced creatures teach less experienced creatures right actions on the right objects. 

I'm not arguing against having imported knowledge. Hell, it can supplement our agent in lots of ways. I imagine how good it would be if an agent would learn about relationships between things from a human knowledge graph. It would be like me reading some secret compendium of knowledge in a Hebrew book. But my intelligence shouldn't be grounded in my reading the Hebrew book, it's grounded in survival and my reading external sources of information is just an incidental consequence of my fitness.

## Generalized Latent Discrimination Capacity 
Indirect knowledge is when our perceptual system is able to differentiate objects that it hadn’t learned about directly. An argument that one might give is that evolution gives creatures mechanisms to distinguish between two things that aren’t relevant to its fitness. For example, human beings are able to distinguish between. 

Perceptual learning and discrimination refers to the ability of an organism (human or animal) to improve in discriminating stimuli via experience, attention, repeated exposure, etc. In human experience, distinguishing objects you had not learned about directly is common, but is a step beyond simple perceptual learning. "Here we suggest that transfer of learning takes place when the trained and untrained stimuli and task activate overlapping brain processes. " ~ KahalaniHodedany2024 So the ability to discriminate or treat new stimuli differently is there, but the mechanism often still involves some structural similarity to learned stimuli. Also of interest is how animals develop fears of certain creatures, eg, snakes, and how these fears generalize?
KahalaniHodedany2024 show how transfer works in human beings for tasks that use the same neural pathways, but no transfer for tasks that use different pathways. But it doesn’t provide a mechanism of how such learning would take place in the first place, ie, how learning is bootstrapped. 

Even if the network has never seen a specific cat or tree, it can still classify them based on shared compositional structure… this ability is also quite obvious in artificial neural networks, where scenes are categorized and given coordinates in feature space… DINO, AlexNet, BYOL, etc. 

Monkeys and snakes: Lab-raised monkeys with no exposure to snakes instantly develop fear after watching another monkey react fearfully to a snake — but not to a flower. (Susan Mineka, 1984)

Put together, this gives us the phenomenon where an organism can differentiate novel stimuli because its perceptual and representational machinery already encodes the right structure to do so. Also evolution doesn’t design rational fears; it guesses useful heuristics and lets reality keep score.

## Fitness as the Only Long-term-relevant Objective 
Often times people optimize for things that are not fitness. 
In them is the implicit assumption of knowledge of the exact relationship between fitness and the non-fitness objective they are optimizing for. 

"Fitness is the only “objective function” that truly matters"

We argue that perception only matters insofar as it serves survival (fitness) — approach nutrients, avoid danger, sustain structure. That doesn't lead to a redefinition of what perceptions is; it only means its relevance to artificial life isn't in representations, but in fitness. 

In robotics, it is often necessary to solve expedient problems - eg, 
- object detection
- semantic segmentation (SAM, SAM2), SLAM (MASt3R-SLAM - Murai2025)
But this sort of thing might be hitting low and failing to achieve real open endedness when applied to artificial life
- maximum coverage
- curiosity 
- behavior novelty 
- exploration 

From the point of view of artificial life requirements, it can be posited that perception isn’t about representing the world, but about maintaining existence within it. I’m thinking that perception can serve fitness in a very major way.

Lehman and Stanley (2011) demonstrated in a set of experiments that using the novelty of a solution instead of the resulting performance on a task can actually lead to much better results. In these experiments, the performance criterion was still used to recognize a good solution when it was discovered, but not to drive the search process. 

## Human Priors 
- Chaplot2020SemanticCuriosity- uses human objects known before hand 
- Yang2019VisualSemantic - prior knowledge of what mangoes are. 

All the same human priors are usually partially right about the requirements for life - eg, they need energy or repair. These are usually obvious. The thing is that the finer requirements are usually subtler, enmeshments between the controller, environment and morphology. 

## Oracle Labelling 
- Oracle labelling - Chaplot2020SemanticCuriosity, 
- "Oracle labeling" is a farce. This is the original sin of modern embodied AI.
- My point: In real life (biological or silicon), no oracle exists. Perception must emerge from interaction + selection pressure, not annotation or oracle knowledge. 

Supervisor provides reward signal (distance to goal) - Yang2019VisualSemantic

## On the Research Process 
### Generalization vs Specialization 
- Doncieux2015EvolutionaryRobotics, Schrodinger, Weiner, Braitenberg

**Problem**
- Modern science fragments knowledge; specialists master narrow domains, losing sight of holistic understanding necessary to synthesize complex systems like life. ~ Schrodinger1944
- Robotics design is fragmented; morphology, sensors, and control interact complexly, yet engineering often treats them in isolation. ~ Doncieux2015EvolutionaryRobotics
- Scientific specialization isolates researchers into narrow domains, causing duplication, lost insights, and barriers to integrating knowledge across related fields. ~ Wiener2019
**Solution**
Venture to integrate diverse knowledge, combining specialized expertise with synthetic perspective, accepting incomplete understanding and potential errors for meaningful insight. ~ Schrodinger1944  

- Evolutionary robotics integrates morphology, sensors, and control holistically, using variation and selection to optimize robot behavior across interdependent components. ~ Doncieux2015EvolutionaryRobotics

Foster interdisciplinary “no-man’s land” collaboration where specialists understand neighboring fields, enabling synthesis exemplified by Wiener’s creation of Cybernetics. ~ Wiener2019
### Synthesis vs Analysis 
- Synthesis vs analysis - Braitenberg1984

Uphill analysis—inferring internal mechanisms from observed behavior—leads to overestimated complexity and indeterminate explanations of underlying structure.

Downhill synthesis—constructing systems from simple components—reveals how complex, lifelike behaviors emerge naturally from minimal sensory-motor architectures.