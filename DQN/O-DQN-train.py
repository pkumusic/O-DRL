#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing, Yuxin

from tensorpack.tfutils import symbolic_functions as symbf

import argparse
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
from tensorpack.RL.expreplay import ExpReplay
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.train.queue import QueueInputTrainer
from tensorpack.RL.common import MapPlayerState, ObjectSensitivePlayer, show_images
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.common import LimitLengthPlayer, PreventStuckPlayer
from tensorpack.RL.history import HistoryFramePlayer
from tensorpack.tfutils.argscope import argscope
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.pool import MaxPooling
from tensorpack.models.nonlin import LeakyReLU, PReLU
from tensorpack.models.fc import FullyConnected
import tensorpack.tfutils.summary as summary
from tensorpack.tfutils.gradproc import MapGradient, SummaryGradient
from tensorpack.callbacks.graph import RunOp
from tensorpack.callbacks.base import PeriodicCallback
import numpy as np


import common
from common import play_model, Evaluator, eval_model_multithread
from obj_recognizor import TemplateMatcher

BATCH_SIZE = 64
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = None #4
ACTION_REPEAT = 4

#CHANNEL = FRAME_HISTORY * 3
IMAGE_SHAPE3 = None #IMAGE_SIZE + (CHANNEL,)
GAMMA = 0.99

INIT_EXPLORATION = 1
EXPLORATION_EPOCH_ANNEAL = 0.01
END_EXPLORATION = 0.1

MEMORY_SIZE = 2e4
# NOTE: will consume at least 1e6 * 84 * 84 bytes == 6.6G memory.
# Suggest using tcmalloc to manage memory space better.
INIT_MEMORY_SIZE = 5e2
STEP_PER_EPOCH = 10#000
EVAL_EPISODE = 1

NUM_ACTIONS = None
DOUBLE = None
DUELING = None
OBJECT_METHOD = None
TEMPLATE_MATCHER = None

def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    global NUM_ACTIONS
    global IMAGE_SHAPE3
    global FRAME_HISTORY
    NUM_ACTIONS = pl.get_action_space().num_actions()
    def resize(img):
        return cv2.resize(img, IMAGE_SIZE)
    def grey(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = resize(img)
        img = img[:, :, np.newaxis] / 255.0
        return img
    if OBJECT_METHOD == 'swap_input_combine': #1
        # swap the input with combined object image
        FRAME_HISTORY = 4
        IMAGE_SHAPE3 = IMAGE_SIZE + (FRAME_HISTORY,)
        pl = ObjectSensitivePlayer(pl, TEMPLATE_MATCHER, OBJECT_METHOD, resize)
        #pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        #show_images(pl.current_state())

    if OBJECT_METHOD == 'add_input_combine': #2
        # add the input with combined object image for each history
        FRAME_HISTORY = 4
        IMAGE_SHAPE3 = IMAGE_SIZE + (FRAME_HISTORY * 2,)
        pl = MapPlayerState(pl, grey)
        pl = ObjectSensitivePlayer(pl, TEMPLATE_MATCHER, OBJECT_METHOD, resize)
        #pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        #show_images(pl.current_state())

    if OBJECT_METHOD == 'add_input_separate': #3
        # For the current state, add the object images
        # (obj1_his1, obj2_his1, cur_his1, obj1_his2, ... )
        # For each image, use the grey scale image, and resize it to 84 * 84
        FRAME_HISTORY = 4
        IMAGE_SHAPE3 = IMAGE_SIZE + (FRAME_HISTORY * (len(TEMPLATE_MATCHER.index2obj)+1),)
        pl = MapPlayerState(pl, grey)
        pl = ObjectSensitivePlayer(pl, TEMPLATE_MATCHER, OBJECT_METHOD, resize)
        #show_images(pl.current_state())

    if OBJECT_METHOD == 'swap_input_separate': #4
        # swap the input images with object images
        # (obj1_his1, obj2_his1, obj1_his2, obj2_his2, ... )
        # TODO: If we need to add walls
        # Interesting thing is we don't have any wall info here
        FRAME_HISTORY = 4
        IMAGE_SHAPE3 = IMAGE_SIZE + (FRAME_HISTORY * len(TEMPLATE_MATCHER.index2obj),)
        pl = ObjectSensitivePlayer(pl, TEMPLATE_MATCHER, OBJECT_METHOD, resize)

    if not train:
        pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        pl = PreventStuckPlayer(pl, 30, 1)
        #show_images(pl.current_state())

    pl = LimitLengthPlayer(pl, 40000)
    #show_images(pl.current_state())
    #exit()
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
        #image = image / 255.0
        #print tf.shape(image)
        with argscope(Conv2D, nl=PReLU.f, use_bias=True):
            l = Conv2D('conv0', image, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool0', l, 2)
            l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
            l = MaxPooling('pool1', l, 2)
            l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
            l = MaxPooling('pool2', l, 2)
            l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

            l = FullyConnected('fc0', l, 512, nl=lambda x, name: LeakyReLU.f(x, 0.01, name))
            # the original arch
            #.Conv2D('conv0', image, out_channel=32, kernel_shape=8, stride=4)
            #.Conv2D('conv1', out_channel=64, kernel_shape=4, stride=2)
            #.Conv2D('conv2', out_channel=64, kernel_shape=3)

        if not DUELING:
            Q = FullyConnected('fct', l, NUM_ACTIONS, nl=tf.identity)
        else:
            V = FullyConnected('fctV', l, 1, nl=tf.identity)
            As = FullyConnected('fctA', l, NUM_ACTIONS, nl=tf.identity)
            Q = tf.add(As, V - tf.reduce_mean(As, 1, keep_dims=True))
        return tf.identity(Q, name='Qvalue')


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

        if not DOUBLE:
            # DQN  # Select the greedy and value from the same target net.
            best_v = tf.reduce_max(targetQ_predict_value, 1)    # N,
        else:
            # Double-DQN # select the greedy from online net, get value from the target net.
            tf.get_variable_scope().reuse_variables()
            next_predict_value = self._get_DQN_prediction(next_state)
            self.greedy_choice = tf.argmax(next_predict_value, 1)   # N,
            predict_onehot = tf.one_hot(self.greedy_choice, NUM_ACTIONS, 1.0, 0.0)
            best_v = tf.reduce_sum(targetQ_predict_value * predict_onehot, 1)

        target = reward + (1.0 - tf.cast(isOver, tf.float32)) * GAMMA * tf.stop_gradient(best_v)

        self.cost = tf.truediv(symbf.huber_loss(target - pred_action_value),
                               tf.cast(BATCH_SIZE, tf.float32), name='cost')

        summary.add_param_summary([('conv.*/W', ['histogram', 'rms']),
                                   ('fc.*/W', ['histogram', 'rms']) ])   # monitor all W

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

# def get_config():
#     logger.auto_set_dir()
#     M = Model()
#
#     lr = tf.Variable(0.001, trainable=False, name='learning_rate')
#     tf.scalar_summary('learning_rate', lr)
#
#     return TrainConfig(
#         #dataset = ?, # A dataflow object for training
#         optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
#         callbacks=Callbacks([StatPrinter(), ModelSaver(),
#
#                              ]),
#
#         session_config = get_default_sess_config(0.6),  # Tensorflow default session config consume too much resources.
#         model = M,
#         step_per_epoch=STEP_PER_EPOCH,
#     )

def get_config():
    #logger.auto_set_dir()
    #logger.set_logger_dir(os.path.join('train_log', LOG_DIR))
    logger.set_logger_dir(LOG_DIR)
    M = Model()

    dataset_train = ExpReplay(
            predictor_io_names=(['state'], ['Qvalue']),
            player=get_player(train=True),
            batch_size=BATCH_SIZE,
            memory_size=MEMORY_SIZE,
            init_memory_size=INIT_MEMORY_SIZE,
            exploration=INIT_EXPLORATION,
            end_exploration=END_EXPLORATION,
            exploration_epoch_anneal=EXPLORATION_EPOCH_ANNEAL,
            update_frequency=4,
            #reward_clip=(-1, 1),
            history_len=FRAME_HISTORY)

    lr = tf.Variable(0.001, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), PeriodicCallback(ModelSaver(), 5),
            ScheduledHyperParamSetter('learning_rate',
                [(100, 4e-4), (500, 1e-4), (1000, 5e-5)]),
            RunOp(lambda: M.update_target_param()),
            dataset_train,
            PeriodicCallback(Evaluator(EVAL_EPISODE, ['state'], ['Qvalue']), 5),
            #HumanHyperParamSetter('learning_rate', 'hyper.txt'),
            #HumanHyperParamSetter(ObjAttrParam(dataset_train, 'exploration'), 'hyper.txt'),
        ]),
        # save memory for multiprocess evaluator
        session_config=get_default_sess_config(0.6),
        model=M,
        step_per_epoch=STEP_PER_EPOCH,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g','--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-l','--load', help='load model')
    parser.add_argument('-e','--env', help='env', required=True)
    parser.add_argument('-t','--task', help='task to perform',
                        choices=['play','eval','train'], default='train')
    parser.add_argument('--double', help='If use double DQN', default='t')
    parser.add_argument('--dueling', help='If use dueling method', default='f')
    parser.add_argument('--logdir', help='output directory', required=True)
    parser.add_argument('--object', help='Method of incorporating object',
                        choices=['add_input_separate', 'add_input_combine',
                                 'swap_input_separate', 'swap_input_combine',
                                 'add_feature_separate', 'add_feature_combine'], default='add_input_separate')
    args=parser.parse_args()
    ENV_NAME = args.env
    LOG_DIR  = args.logdir
    OBJECT_METHOD = args.object
    TEMPLATE_MATCHER = TemplateMatcher('../obj/MsPacman-v0')

    logger.info('Using Object Method: ' + OBJECT_METHOD)

    if args.double == 't':
        DOUBLE = True
    elif args.double == 'f':
        DOUBLE = False
    else:
        logger.error("double argument must be t or f")
        exit()
    if args.dueling == 't':
        DUELING = True
    elif args.dueling == 'f':
        DUELING = False
    else:
        logger.error("dueling argument must be t or f")
        exit()

    if DOUBLE:
        logger.info("Using Double")
    if DUELING:
        logger.info("Using Dueling")

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
                output_var_names=['Qvalue'])
        if args.task == 'play':
            play_model(cfg)
        elif args.task == 'eval':
            eval_model_multithread(cfg, EVAL_EPISODE)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        QueueInputTrainer(config).train()
