#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing, Yuxin

import gym
import numpy as np
import tensorflow as tf
import os, sys, re, time
import random
import argparse
import subprocess
import multiprocessing, threading
from collections import deque

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.tfutils import symbolic_functions as symbf
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.RL import *
from tensorpack.predict.common import PredictConfig
from tensorpack.models.model_desc import ModelDesc
from tensorpack.train.config import TrainConfig
from tensorpack.tfutils.common import *
from tensorpack.callbacks.group import Callbacks
from tensorpack.callbacks.stat import StatPrinter
from tensorpack.callbacks.common import ModelSaver
from tensorpack.callbacks.param import ScheduledHyperParamSetter, HumanHyperParamSetter

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
from tensorpack.tfutils.argscope import argscope
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.pool import MaxPooling
from tensorpack.models.fc import FullyConnected
from tensorpack.models.nonlin import LeakyReLU

import common
from common import play_model, Evaluator, eval_model_multithread, play_one_episode


# STEP_PER_EPOCH = 6000


BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
ACTION_REPEAT = 4

CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)
GAMMA = 0.99

INIT_EXPLORATION = 1
EXPLORATION_EPOCH_ANNEAL = 0.01
END_EXPLORATION = 0.1

MEMORY_SIZE = 1e6
INIT_MEMORY_SIZE = 5e4
STEP_PER_EPOCH = 1
EVAL_EPISODE = 50

NUM_ACTIONS = None
ROM_FILE = None


def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    def func(img):
        return cv2.resize(img, IMAGE_SIZE[::-1])
    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()
    if not train:
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl
common.get_player = get_player  # so that eval functions in common can use the player

class Model(ModelDesc):
    def _get_input_vars(self):
        if NUM_ACTIONS is None:
            p = get_player(); del p
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state'),
                InputVar(tf.int64, (None,), 'action'),
                InputVar(tf.float32, (None,), 'reward'),
                InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'next_state'),
                InputVar(tf.bool, (None,), 'isOver') ]

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
        with argscope(Conv2D, nl=PReLU.f, use_bias=True):
            return (LinearWrap(image)
                .Conv2D('conv0', out_channel=32, kernel_shape=5)
                .MaxPooling('pool0', 2)
                .Conv2D('conv1', out_channel=32, kernel_shape=5)
                .MaxPooling('pool1', 2)
                .Conv2D('conv2', out_channel=64, kernel_shape=4)
                .MaxPooling('pool2', 2)
                .Conv2D('conv3', out_channel=64, kernel_shape=3)

                # the original arch
                #.Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
                #.Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
                #.Conv2D('conv2', out_channel=64, kernel_shape=3)

                .FullyConnected('fc0', 512, nl=lambda x, name: LeakyReLU.f(x, 0.01, name))
                .FullyConnected('fct', NUM_ACTIONS, nl=tf.identity)())

    def _build_graph(self, inputs):
        state, action, reward, next_state, isOver = inputs
        self.predict_value = self._get_DQN_prediction(state)
        action_onehot = tf.one_hot(action, NUM_ACTIONS, 1.0, 0.0)
        pred_action_value = tf.reduce_sum(self.predict_value * action_onehot, 1)    #N,
        max_pred_reward = tf.reduce_mean(tf.reduce_max(
            self.predict_value, 1), name='predict_reward')
        add_moving_summary(max_pred_reward)

        with tf.variable_scope('target'):
            targetQ_predict_value = self._get_DQN_prediction(next_state)    # NxA
        # DQN
        #best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        # Double-DQN
        tf.get_variable_scope().reuse_variables()
        next_predict_value = self._get_DQN_prediction(next_state)
        self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
        predict_onehot = tf.one_hot(self.greedy_choice, NUM_ACTIONS, 1.0, 0.0)
        best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)
        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * GAMMA * tf.stop_gradient(best_v)
        cost = symbf.huber_loss(target - pred_action_value)
        add_param_summary([('conv.*/W', ['histogram', 'rms']),
                           ('fc.*/W', ['histogram', 'rms']) ])   # monitor all W

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        with tf.name_scope('param_summary'):
            for p in params:
                name = p.name
                print name
                if name == 'conv0/W:0':
                    print "enter ! "
                    weights = tf.reshape(p, [-1,5,5,1])
                    tf.image_summary(name, weights)

        self.cost = tf.reduce_mean(cost, name='cost')

    def update_target_param(self):
        vars = tf.trainable_variables()
        ops = []
        for v in vars:
            target_name = v.op.name
            if target_name.startswith('target'):
                new_name = target_name.replace('target/', '')
                logger.info("{} <- {}".format(target_name, new_name))
                ops.append(v.assign(tf.get_default_graph().get_tensor_by_name(new_name + ':0')))
        return tf.group(*ops, name='update_target_network')

    def get_gradient_processor(self):
        return [MapGradient(lambda grad: \
                tf.clip_by_global_norm([grad], 5)[0][0]),
                SummaryGradient()]


def get_config():
    logger.auto_set_dir()
    M = Model()

    dataset_train = ExpReplay(
            predictor_io_names=(['state'], ['fct/output']),
            player=get_player(train=True),
            batch_size=BATCH_SIZE,
            memory_size=MEMORY_SIZE,
            init_memory_size=INIT_MEMORY_SIZE,
            exploration=INIT_EXPLORATION,
            end_exploration=END_EXPLORATION,
            exploration_epoch_anneal=EXPLORATION_EPOCH_ANNEAL,
            update_frequency=4,
            reward_clip=(-1, 1),
            history_len=FRAME_HISTORY)

    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            ScheduledHyperParamSetter('learning_rate',
                [(150, 4e-4), (250, 1e-4), (350, 5e-5)]),
            RunOp(lambda: M.update_target_param()),
            dataset_train,
            PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['fct/output']), 3),
            #HumanHyperParamSetter('learning_rate', 'hyper.txt'),
            #HumanHyperParamSetter(ObjAttrParam(dataset_train, 'exploration'), 'hyper.txt'),
        ]),
        # save memory for multiprocess evaluator
        session_config=get_default_sess_config(0.6),
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
    )


def run_submission(cfg, output, nr):
    player = get_player(dumpdir=output)
    predfunc = get_predict_func(cfg)
    for k in range(nr):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)
        print("Total:", score)
    player.close()

def do_submit(output, api_key):
    gym.upload(output, api_key=api_key)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-l','--load', help='load model')
    parser.add_argument('-e','--env', help='env', required=True)
    parser.add_argument('-t','--task', help='task to perform',
                        choices=['play','eval','run','train'], default='train')
    parser.add_argument('-p','--episode', help='number of episodes to run',
            type=int, default=1)
    parser.add_argument('-o','--output', help='output directory', default='gym-submit')
    parser.add_argument('-k','--key', help='api key')
    args=parser.parse_args()
    ENV_NAME = args.env
    assert ENV_NAME
    p = get_player(); del p     # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.task != 'train':
        assert args.load is not None

    if args.task != 'train':
        cfg = PredictConfig(
                model=Model(),
                session_init=SaverRestore(args.load),
                input_var_names=['state'],
                output_var_names=['fct/output:0'])
        if args.task == 'play':
            play_model(cfg)
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE)
        elif args.task == 'run':
            run_submission(cfg, args.output, args.episode)
            do_submit(args.output, args.key)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
