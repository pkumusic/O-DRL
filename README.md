# O-DRL
Object Sensitive Deep Reinforcement Learning

# TODO List for the Deep Learning Project (Yuezhang, Tian, Jing) 
+ Choose the game we wanna use. (Having different objects) (Ranked by preference for experiments) (Done)
    * Freeway-v0: A chicken is trying to across the street while avoid hitting by cars 
    * MsPacman-v0: Pacman eat beans, pellets, and may eat monsters. 
    * Riverraid-v0: Has fuel and ship objects which have opposite effects.
    * Berzerk-v0: Walls, bullets and enemies to avoid, enemies to kill.
    * Pooyan-v0: The game I love when I was a child. Some objects inside. 
    * (Hard) MontezumaRevenge-v0: A very hard game for DRL where we need to get keys and its a long adventure game. (So the reward is delayed and sparse)
    * (Hard) Kangaroo-v0: A3C seems performs bad. Has several objects.
    * (Hard) Skiing-v0: The skier needs to across the areas inside the two flags to get scores.
+ Train the A3C model for the game and evaluate it on gym. (Done)
`./train-atari.py --env MsPacman-v0 --gpu 0`
+ Rewrite the DQN to fit in the gym env
+ Visualize and evaluate the hidden features of the network to see if it encodes any object/edge info.
+ Apply pre-trained edge/object detection techniques, combining it to A3C model.
+ Can we incorporate the object/edge detection objective to the DRL model?


# Git Configuration
Initialization
~~~
https://github.com/pkumusic/O-DRL.git
~~~


# Deploy on Mac
+ I did it without virtual environment. If using machines without root privilege, it maybe more convenient to use virtualenv 
+ Python 2 or 3
+ Install TensorFlow >= 0.10 
```pip install tensorflow```
Please do this before installing numpy. Since sometimes numpy would have version confliction with tensorflow version.
+ Python bindings for OpenCV
``` pip install opencv-python```
+ other requirements: (Please refer to the txt file below)
```
pip install --user -r requirements.txt
pip install --user -r opt-requirements.txt (some optional dependencies, you can install later if needed)
```
+ Enable `import tensorpack`:
```
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
```

# Additional on deploy on Tian's Linux Machine
1. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
2. 
~~~~
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
export PYTHONPATH=$PYTHONPATH:`readlink -f /usr0/home/ttian1/ml/tensorpack`
~~~~


# tensorpack
Neural Network Toolbox on TensorFlow

See some [examples](examples) to learn about the framework.
You can actually train them and reproduce the performance... not just to see how to write code.

+ [DoReFa-Net: training binary / low bitwidth CNN](examples/DoReFa-Net)
+ [InceptionV3 on ImageNet](examples/Inception/inceptionv3.py)
+ [ResNet for Cifar10 classification](examples/ResNet)
+ [Fully-convolutional Network for Holistically-Nested Edge Detection](examples/HED)
+ [Spatial Transformer Networks on MNIST addition](examples/SpatialTransformer)
+ [Double DQN plays Atari games](examples/Atari2600)
+ [Asynchronous Advantage Actor-Critic(A3C) with demos on OpenAI Gym Atari games](examples/OpenAIGym)
+ [char-rnn language model](examples/char-rnn)

## Features:

Describe your training task with three components:

1. Model, or graph. `models/` has some scoped abstraction of common models, but you can simply use
	 anything available in tensorflow. This part is roughly an equivalent of slim/tflearn/tensorlayer.
	`LinearWrap` and `argscope` makes large models look simpler.

2. Data. tensorpack allows and encourages complex data processing.

	+ All data producer has an unified `generator` interface, allowing them to be composed to perform complex preprocessing.
	+ Use Python to easily handle any of your own data format, yet still keep a good training speed thanks to multiprocess prefetch & TF Queue prefetch.
	For example, InceptionV3 can run in the same speed as the official code which reads data using TF operators.

3. Callbacks, including everything you want to do apart from the training iterations. Such as:
	+ Change hyperparameters during training
	+ Print some variables of interest
	+ Run inference on a test dataset
	+ Run some operations once a while
	+ Send the accuracy to your phone

With the above components defined, tensorpack trainer will run the training iterations for you.
Multi-GPU training is off-the-shelf by simply switching the trainer.

## Dependencies:

+ Python 2 or 3
+ TensorFlow >= 0.10
+ Python bindings for OpenCV
+ other requirements:
```
pip install --user -r requirements.txt
pip install --user -r opt-requirements.txt (some optional dependencies, you can install later if needed)
```
+ Use [tcmalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) whenever possible
+ Enable `import tensorpack`:
```
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
```
