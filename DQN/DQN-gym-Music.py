#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
import argparse
import os
from tensorpack.predict.common import PredictConfig
from tensorpack import *
from tensorpack.models.model_desc import ModelDesc, InputVar
from tensorpack.train.config import TrainConfig
from tensorpack.tfutils.common import *
from tensorpack.callbacks.group import Callbacks
from tensorpack.callbacks.stat import StatPrinter
from tensorpack.callbacks.common import ModelSaver
from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.symbolic_functions import huber_loss
from tensorpack.RL.expreplay import ExpReplay
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.trainer import QueueInputTrainer
from tensorpack.RL.common import MapPlayerState
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.common import LimitLengthPlayer, PreventStuckPlayer
from tensorpack.RL.history import HistoryFramePlayer
import common
from tensorpack.tfutils.argscope import argscope
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.pool import MaxPooling
from tensorpack.models.fc import FullyConnected
from tensorpack.models.nonlin import LeakyReLU
STEP_PER_EPOCH = 6000

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
IMAGE_SHAPE3 = IMAGE_SIZE + (FRAME_HISTORY, ) # one state input


NUM_ACTIONS = None

GAMMA = 0.99

def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    def func(img):
        return cv2.resize(img, IMAGE_SIZE[::-1]) #TODO: Do we really need to resize here? Check the original paper.
    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()

    if not train: # When testing
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        #pl = PreventStuckPlayer(pl, 30, 1) #TODO: Need to know the start button. Is it different for each game?
    pl = LimitLengthPlayer(pl, 30000) # 500s
    return pl
common.get_player = get_player()

class Model(ModelDesc):
    def _get_input_vars(self):
        if NUM_ACTIONS is None:
            p = get_player(); del p
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'reward'),
                InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'next_state'),
                InputVar(tf.bool, (None,), 'isOver')]

    def _get_DQN_prediction(self, image):
        #TODO: Do we need to add other pre-processing? e.g., subtract mean
        image = image / 255.0
        #TODO: The network structure can be improved?
        with argscope(Conv2D, nl=tf.nn.relu, use_bias=True): # Activation for each layer
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=3)
            # the original arch
            # .Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
            # .Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
            # .Conv2D('conv2', out_channel=64, kernel_shape=3)
            l = FullyConnected('fc0', l, 512, nl=lambda x, name:LeakyReLU.f(x, 0.01, name))
            l = FullyConnected('fct', l, NUM_ACTIONS, nl=tf.identity())

    def _build_graph(self, inputs):
        state, action, reward, next_state, isOver = inputs
        predict_value = self._get_DQN_prediction() # N * NUM_ACTIONS #TODO: If we need self. here
        action_onehot = tf.one_hot(action, NUM_ACTIONS, 1.0, 0.0) # N * NUM_ACTION
        pred_action_value = tf.reduce_sum(predict_value * action_onehot, 1) # N,

        ### This is for tracking the learning process.
        # The mean max-Q across samples. Should be increasing over training
        max_pred_reward = tf.reduce_mean(tf.reduce_max(predict_value, 1),
                             name='predict_reward')
        add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'): #TODO: Check the usage of variable scope in this context
            targetQ_predict_value = self._get_DQN_prediction(next_state)

        # DQN
        best_v = tf.reduce_max(targetQ_predict_value, 1)

        #TODO: Double-DQN

        #TODO: Why we need stop_gradient here
        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * GAMMA * tf.stop_gradient(best_v)

        cost = huber_loss(target - pred_action_value)

        add_param_summary([('conv.*/W', ['histogram', 'rms']),
                           ('fc.*/W', ['histogram', 'rms'])]) #TODO
        self.cost = tf.reduce_mean(cost, name='cost')

def get_config():
    logger.auto_set_dir()
    M = Model()
    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    dataset_train = ExpReplay()

    return TrainConfig(
        dataset=dataset_train, # A dataflow object for training
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([StatPrinter(), ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',[(80, 0.0003), (120, 0.0001)]) # No interpolation
            # TODO: Some other parameters

            ]),

        session_config = get_default_sess_config(0.6),  # Tensorflow default session config consume too much resources.
        model = M,
        step_per_epoch=STEP_PER_EPOCH,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gpu', help='comma seperated list of GPU(s) to use.')
    parser.add_argument('-l','--load', help='load model')
    parser.add_argument('-e','--env', help='env', required=True)
    parser.add_argument('-t','--task', help='task to perform',
                        choices=['play','eval','train'], default='train')
    args=parser.parse_args()
    ENV_NAME = args.env

    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.task == 'train':
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
    else:
        pass
