# O-DRL
Object Sensitive Deep Reinforcement Learning. Combining Object Features with Deep Reinforcement Learning methods.

This implementation of Deep Reinforcement Learning (DRL) methods are based on [tensorflow](https://www.tensorflow.org/) and [tensorpack](https://github.com/ppwwyyxx/tensorpack). I used an older version of tensorpack and did some changes to fit into our project. For installing dependencies of tensorpack, please refer [here](https://github.com/ppwwyyxx/tensorpack).

# Some baselines in OpenAI Gym
+ [DQN](https://gym.openai.com/evaluations/eval_riHeA66BRujxsfQHz6FDA): Nature DQN in [Human-level control through deep reinforcement
learning](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf)
+ [DDQN](https://gym.openai.com/evaluations/eval_3XssszDyStmC4tkhx8wdcg): Double DQN in [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
+ [Dueling](https://gym.openai.com/evaluations/eval_6fKnCcOTqOxnbW0JwJLTw): Dueling DQN in [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)
+ [DDDQN](https://gym.openai.com/evaluations/eval_7UlV9il3RFKNyciZOVu2g): One incorporating Double and Dueling DQN 

# Git Configuration
Initialization

```
https://github.com/pkumusic/O-DRL.git
```

export path 

```
export PYTHONPATH=$PYTHONPATH:path/to/O-DRL
```

# Deploy on Mac
## Dependencies of Tensorpack
+ please refer [tensorpack](https://github.com/ppwwyyxx/tensorpack) for installation. Below is what I did in my Mac.
+ Python 2 or 3
+ Install TensorFlow >= 0.10 
```pip install tensorflow```
Please do this before installing numpy. Since sometimes numpy would have version confliction with tensorflow version.
+ Python bindings for OpenCV ``` pip install opencv-python``` (This works only for CPU mode. If you're using GPU, please compile opencv with Python executable enabled. [reference](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html))
+ Installing requirements for tensorpack

# Additional on deploy on our turtle Machine
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64`
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
export PYTHONPATH=$PYTHONPATH:`readlink -f /usr0/home/ttian1/ml/tensorpack`
```
