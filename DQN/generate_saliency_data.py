#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music
from __future__ import division
import argparse
from tensorpack.predict.common import PredictConfig
from tensorpack import *
from tensorpack.models.model_desc import ModelDesc, InputVar
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.RL.common import MapPlayerState
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.history import HistoryFramePlayer
from tensorpack.tfutils.argscope import argscope
from tensorpack.models.conv2d import Conv2D
from tensorpack.models.pool import MaxPooling
from tensorpack.models.nonlin import LeakyReLU, PReLU
from tensorpack.models.fc import FullyConnected
from tensorpack.predict.base import OfflinePredictor
from saliency_analysis import Saliency_Analyzor
from obj_recognizor import TemplateMatcher
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import pickle
#from matplotlib_colorbar.colorbar import Colorbar

IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4
CHANNEL = FRAME_HISTORY# * 3
IMAGE_SHAPE3 = IMAGE_SIZE + (CHANNEL,)

NUM_ACTIONS = None
ENV_NAME = None
DOUBLE = None
DUELING = None

Action_Dict = {0:'nowhere', 1:'up', 2:'right', 3:'left', 4:'down', 5:'upright', 6:'upleft', 7:'downright', 8:'downleft'}

from common import play_one_episode, get_predict_func

def get_player(dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir, auto_restart=False)
    pl = MapPlayerState(pl, grey)
    pl = MapPlayerState(pl, resize)
    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()
    pl = HistoryFramePlayer(pl, FRAME_HISTORY)
    #show_images(pl.current_state())
    return pl

def show_images(img, last=False, grey=True):
    # util function for showing images
    import matplotlib.pyplot as plt
    if grey:
        plt.imshow(img)
        plt.show()
        return
    for i in xrange(img.shape[2]):
        if last:
            if i != img.shape[2]-1:
                continue
        plt.imshow(img[:,:,i])
        plt.show()


def resize(img):
    img = cv2.resize(img, IMAGE_SIZE)
    img = img[:, :, np.newaxis]
    return img

def grey(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img[:, :, np.newaxis]
    return img

def expand_state(s, need_grey=True):
    s = grey(s, need_grey)
    s = [s,s,s,s]
    s = np.concatenate(s, axis=2)
    return s

class Model(ModelDesc):
    def _get_input_vars(self):
        assert NUM_ACTIONS is not None
        return [InputVar(tf.float32, (None,) + IMAGE_SHAPE3, 'state')]#,
                #InputVar(tf.int32, (None,), 'action'),
                #InputVar(tf.float32, (None,), 'futurereward') ]

    def _get_DQN_prediction(self, image):
        """ image: [0,255]"""
        image = image / 255.0
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
        state = inputs[0]
        #state, action, futurereward = inputs
        self.Qvalue = self._get_DQN_prediction(state)
        max_Qvalue = tf.reduce_max(self.Qvalue, 1)
        saliency = tf.gradients(max_Qvalue, state)[0]
        self.saliency = tf.identity(saliency, name='saliency')


def get_history_state(history):
    assert len(history) != 0
    diff_len = history.maxlen - len(history)
    if diff_len == 0:
        return np.concatenate(history, axis=2)
    zeros = [np.zeros_like(history[0]) for k in range(diff_len)]
    for k in history:
        zeros.append(k)
    assert len(zeros) == history.maxlen
    return np.concatenate(zeros, axis=2)

def sample_epoch_for_analysis(predfunc, s_func, output):
    """
    :param cfg: cfg to predict Q values
    :param s_cfg: cfg to predict pixel saliency maps in 84 * 84 * 4
    :param output: output folder name
    :return:  save the sampled epoch arrays in output folder
    Arrays including (original_state(210*160*3), unresized_states(210*160*4), states(84*84*4), saliency(84*84*4), act, r, timestep)
    """
    player = get_player(dumpdir=output)
    timestep = 0
    sa = Saliency_Analyzor('../obj/MsPacman-v0')
    R = 0
    history = deque(maxlen=FRAME_HISTORY)
    while True:
        timestep += 1
        s0 = player.original_current_state()
        unresized_state = grey(s0)
        history.append(unresized_state)
        us = get_history_state(history)
        # Try to use four save frames to predict action
        s = player.current_state()
        # s = expand_state(s0)
        # Actions: 0-none; 1-Up; 2-right; 3-left; 4-down; 5-upright,6-leftup, 7-rightdown, 8-leftdown
        Qvalues = predfunc([[s]])[0][0]
        act = Qvalues.argmax()
        saliency = s_func([[s]])[0][0]
        #description = generate_description(Qvalues, act)
        r, isOver = player.action(act)
        if isOver:
            history.clear()
        save_arrays(s0, us, s, saliency, act, r, timestep, output)
        #show(s, saliency, act, timestep, output, last=True, save=True)
        #show_large(s0, saliency, act, timestep, output, save=True, save_npy=False, analyzor=sa, description=description, explanation=True)
        #print r, act
        R += r
        if timestep % 50 == 0:
            print timestep
            print 'Total Reward:', R
        if isOver:
            return

def save_arrays(s0, us, s, saliency, act, r, timestep, output):
    np.savez(output + "/arrays%d" % timestep, s0=s0, us=us, s=s, saliency=saliency, act=act, r=r)
    return

def change_points(tm, predfunc):
    Q_values = []
    variances = []
    for i in xrange(1,1291):
        arrays = np.load('arrays2/arrays%d.npz' % i)
        s0, us, s, saliency, act, r = arrays['s0'], arrays['us'], arrays['s'], arrays['saliency'], int(
            arrays['act']), float(arrays['r'])
        Qvalues = predfunc([[s]])[0][0]
        Qvalue = Qvalues[act]
        variance = np.ptp(Qvalues)
        variances.append(variance)
        #Q_values.append(Qvalue)
    highlight(variances)
    #pickle.dump(Q_values, open('seq2/Q_values', 'w'))

def change_points_saliency(tm, predfunc):
    diff = []
    for i in xrange(1, 1291):
        arrays1 = np.load('arrays2/arrays%d.npz' % i)
        saliency1 = np.array(arrays1['saliency'])
        arrays2 = np.load('arrays2/arrays%d.npz' % (i+1))
        saliency2 = np.array(arrays2['saliency'])
        dist = np.linalg.norm(saliency1 - saliency2)
        diff.append(dist)
    highlight(diff)




def highlight(values):
    #diff = [abs(values[i] - values[i-1]) for i in xrange(1, len(values))]
    #change_points = [i+2 for i in xrange(len(diff)) if diff[i] > 15]
    change_points = [i for i in xrange(len(values)) if values[i] > 15]
    print change_points
    plt.plot(values)
    plt.plot()
    plt.show()
    print values




def real_act(tm, output):
    start = 1
    arrays = np.load(output+'/arrays%d.npz' % start)
    s0, us, s, saliency, act, r = arrays['s0'], arrays['us'], arrays['s'], arrays['saliency'], int(
        arrays['act']), float(arrays['r'])
    extracted_objects = tm.match_all_objects(s0)
    positions = [p for p in extracted_objects['pacman'] if not (
    (p.left == 27 and p.right == 37 and p.up == 172 and p.down == 184) or (
    p.left == 11 and p.right == 21 and p.up == 172 and p.down == 184))]
    if positions:
        last_position = positions[0]
    else:
        last_position = None
    last_act = 'none'
    real_acts = {}
    change_points=[]
    for index in xrange(2, 1291):
        arrays = np.load(output+'/arrays%d.npz' % index)
        s0, us, s, saliency, act, r = arrays['s0'], arrays['us'], arrays['s'], arrays['saliency'], int(arrays['act']), float(arrays['r'])
        extracted_objects = tm.match_all_objects(s0)
        positions = [p for p in extracted_objects['pacman'] if not ((p.left==27 and p.right==37 and p.up==172 and p.down==184) or (p.left==11 and p.right==21 and p.up==172 and p.down==184))]
        if positions:
            cur_position = positions[0]
        else:
            cur_position = None
        if index <= 86:
            real_act = 'none'
        elif cur_position is None or last_position is None:
            real_act = last_act
        elif cur_position.up < last_position.up and cur_position.down < last_position.down:
            real_act = 'up'
        elif cur_position.up > last_position.up and cur_position.down > last_position.down:
            real_act = 'down'
        elif cur_position.left > last_position.left and cur_position.right > last_position.right:
            real_act = 'right'
        elif cur_position.left < last_position.left and cur_position.right < last_position.right:
            real_act = 'left'
        elif cur_position.left == last_position.left and cur_position.right == last_position.right and cur_position.up == last_position.up and cur_position.down == last_position.down:
            real_act = 'none'
        else:
            real_act = last_act
        real_acts[index-1] = real_act
        if real_act != last_act:
            change_points.append(index)
        last_act = real_act
        last_position = cur_position
    #print change_points
    pickle.dump(real_acts, open(output+'/real_acts', 'w'))
    #pickle.dump(change_points, open('seq2/real_acts_change_points', 'w'))


def object_saliencies(index, predfunc, s_func, tm, draw=False, save=False):
    """
    Produce object saliencies for each object.
    :return: [(saliency, obj, Position),...,]. obj can be used to find x_len and y_len
    """
    # for a given array, iterate through its all objects,
    # and calculate obj saliency for each by masking out the object to
    # see how much it changed for the Q-value of the given action.
    arrays = np.load('array-O-DDQN/arrays%d.npz' % index)
    #s0, us, s, saliency, act, r = arrays['s0'], arrays['us'], arrays['s'], arrays['saliency'], int(arrays['act']), float(arrays['r'])
    s0, us, s = arrays['s0'], arrays['us'], arrays['s']
    pixel_saliency = s_func([[s]])[0][0]
    pixel_saliency = pixel_saliency[:,:,0]
    pixel_saliency = cv2.resize(pixel_saliency, (160, 210), interpolation=cv2.INTER_NEAREST)
    #print pixel_saliency.shape
    #plt.imshow(pixel_saliency, cmap='gray')
    #plt.show()
    # detect objects in s0
    #mask_wall(us)
    extracted_objects = tm.match_all_objects(s0)
    Qvalues = predfunc([[s]])[0][0]
    act = Qvalues.argmax()
    Qvalue = Qvalues[act]
    # tm.draw_extracted_image(state, extracted_objects)
    obj_sals = []
    for obj, locs in extracted_objects.iteritems():
        for loc in locs:
            masked_s = mask_with_blank(us, loc, tm)
            masked_Qvalue = predfunc([[masked_s]])[0][0][act]
            diff = masked_Qvalue - Qvalue
            obj_sals.append((diff, obj, loc))
    if draw:
        # Draw the object saliency image
        canvas = np.zeros((210, 160))
        for sal, _, loc in obj_sals:
            canvas[loc.up: loc.down, loc.left: loc.right] = sal
        #canvas = cv2.resize(canvas, (160,210), interpolation = cv2.INTER_NEAREST)
        # Original Image
        plt.subplot(131)
        plt.axis('off')
        fig = plt.imshow(s0, aspect='equal')
        plt.title(Action_Dict[act] + ',' + str(Qvalue) )
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # Object Saliency
        plt.subplot(132)
        plt.axis('off')
        fig = plt.imshow(canvas, cmap='gray', aspect='equal')
        plt.title('Object Saliency Map')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        #colorbar = Colorbar(fig, location='upper left')
        #colorbar.set_ticks([-100,-50,0,50,100])
        #plt.gca().add_artist(colorbar)
        # Pixel Saliency
        plt.subplot(133)
        plt.axis('off')
        fig = plt.imshow(pixel_saliency, cmap='gray', aspect='equal')
        plt.title('Pixel Saliency Map')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        #colorbar = Colorbar(fig, location='upper left')
        #colorbar.set_ticks([-0.03,0,0.03])
        #plt.gca().add_artist(colorbar)
        plt.savefig('models/O-DDQN/' + str(index), bbox_inches='tight', pad_inches=0)
        plt.close()
        #plt.show()
        #plt.close()
    if save:
        # Save obj_sals, act, real_act, Q-value
        pass


    return act

def mask_with_blank(us, loc, tm, background=57):
    #show_images(us)
    masked_us = us.copy()
    masked_us[loc.up: loc.down, loc.left: loc.right, :] = background
    #show_images(masked_us)
    masked_us = cv2.resize(masked_us, IMAGE_SIZE)
    return masked_us

def mask_wall(state, x=50, y=50, background_color=57, wall_color=0):
    print state
    exit()


def generate_description(act):
    action_string = Action_Dict[act]
    description = "Pacman choose to go " + action_string
    return description

def generate_pos_neg_explanation(pos_obj_sals, neg_obj_sals):
    s = 'Because the system is aware of: \n'
    for pos_obj in pos_obj_sals:
        saliency, obj, position = pos_obj
        x = (position.left + position.right) / 2
        y = (position.up + position.down) / 2
        s += '%s in (%.1f, %.1f) with saliency value %.1f \n'%(obj, x, y, saliency)
    for neg_obj in neg_obj_sals:
        saliency, obj, position = neg_obj
        x = (position.left + position.right) / 2
        y = (position.up + position.down) / 2
        s += '%s in (%.1f, %.1f) with saliency value %.1f \n' % (obj, x, y, saliency)
    return s

def show_large(s, saliency, act, timestep, output, save=False, save_npy=False, analyzor=None, description=None, explanation=False):
    # Show the pictures of original resolution of the game play
    # Convert the 84*84 saliency maps to 210 * 160 resolution
    import matplotlib.pyplot as plt
    # Get the saliency map for the last frame in the history (The current frame)
    saliency = saliency[:,:,3]
    saliency = cv2.resize(saliency, (160,210))
    if save_npy:
        np.save(output+"/state%d"%timestep, s)
        np.save(output+"/saliency%d"%timestep, saliency)
    # object saliency maps
    if analyzor:
        obj_sals = analyzor.object_saliencies(s, saliency)
        pos_obj_sals, neg_obj_sals = analyzor.object_saliencies_filter(obj_sals)
        s = analyzor.saliency_image(s, pos_obj_sals, neg_obj_sals)
        if explanation:
            explanation = generate_pos_neg_explanation(pos_obj_sals, neg_obj_sals)
    title = ''
    if description:
        title += str(description) + '\n'
    if explanation:
        title += explanation
    plt.subplot(211)
    if description or explanation:
        plt.title(title, fontsize=10)
    plt.axis('off')
    fig = plt.imshow(s, aspect='equal')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(212)
    plt.axis('off')
    fig = plt.imshow(saliency, cmap='gray', aspect='equal')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(output + "/file%04d.png" % timestep, bbox_inches='tight', pad_inches = 0)
    else:
        plt.show()

def show(s, saliency, act, timestep, output, last=False, save=False):
    import matplotlib.pyplot as plt
    for i in xrange(s.shape[2]):
        if last:
            if i != s.shape[2] - 1:
                continue
        plt.subplot(211)
        plt.axis('off')
        fig = plt.imshow(s[:, :, i], cmap='gray', aspect='equal', interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.subplot(212)
        plt.title('action:' + str(act))
        plt.axis('off')
        fig = plt.imshow(saliency[:,:,i], cmap='gray', aspect='equal', interpolation='nearest')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        if save:
            plt.savefig(output + "/file%04d.png" % timestep, bbox_inches='tight', pad_inches = 0)
        else:
            plt.show()

def generate_explanation(obj_sals):
    s = 'Because the pacman is '
    for obj_sal in obj_sals:
        saliency, obj, position = obj_sal
        x = (position.left + position.right) / 2
        y = (position.up + position.down) / 2
        purpose = ''
        if obj == 'dot':
            purpose = 'chasing the ' + obj
        if obj == 'ghost':
            purpose = 'avoiding the ' + obj
        s += purpose
        s += ' in (%.0f, %.0f) with saliency value %.1f \n' % (x, y, abs(saliency))
    return s

def analyze(input, output):
    i = 0
    sa = Saliency_Analyzor('../obj/MsPacman-v0')
    while True:
        i += 1
        arrays = np.load(input + '/' + 'arrays%d.npz' %i)
        s0, s, saliency, act, r = arrays['s0'], arrays['s'], arrays['saliency'], int(arrays['act']), float(arrays['r'])
        desc = generate_description(act)
        saliency = saliency[:, :, 3]
        saliency = cv2.resize(saliency, (160, 210))
        obj_sals = sa.object_saliencies(s0, saliency)
        obj_sals = sa.top_saliency_filter(obj_sals)
        exp = generate_explanation(obj_sals)
        sal_img = sa.saliency_image(s0, obj_sals, obj_sals)
        show_analyze(output, desc, exp, s0, saliency, sal_img, i)

def show_analyze(output, description, explanation, s, saliency, sal_img, timestep, save=True):
    import matplotlib.pyplot as plt
    title = ''
    title += str(description) + '\n'
    title += explanation
    plt.subplot(211)
    if description or explanation:
        plt.title(title, fontsize=10)
    plt.axis('off')
    fig = plt.imshow(sal_img, aspect='equal')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.subplot(212)
    plt.axis('off')
    fig = plt.imshow(saliency, cmap='gray', aspect='equal')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    if save:
        plt.savefig(output + "/file%04d.png" % timestep, bbox_inches='tight', pad_inches = 0)
    else:
        plt.show()

def sensitivity_analysis(index, s_cfg, cfg):
    arrays = np.load('analysis/arrays%d.npz' %index)
    sa = Saliency_Analyzor('../obj/MsPacman-v0')
    s0, s, saliency, act, r = arrays['s0'], arrays['s'], arrays['saliency'], int(arrays['act']), float(arrays['r'])
    # exp = generate_explanation(obj_sals)
    # Mask the object with ghost
    # plt.imshow(saliency, cmap='gray', aspect='equal')
    saliency = saliency[:, :, 3]
    saliency = cv2.resize(saliency, (160, 210))

    obj_sals = sa.object_saliencies(s0, saliency)
    obj_sals = sa.top_saliency_filter(obj_sals)
    saliency, obj, position = obj_sals[0]
    x, y = (position.left + position.right) / 2, (position.up + position.down) / 2
    masked_s0 = mask(s0, x, y, 'ghost', sa)
    #show_images(masked_s0, gray=True)
    # Calculate new saliency for masked image
    s_func = OfflinePredictor(s_cfg)
    #predfunc = OfflinePredictor(cfg)
    #masked_saliency =  s_func([[expand_state(masked_s0, need_grey=False)]])[0][0]
    saliency = s_func([[expand_state(masked_s0, need_grey=False)]])[0][0]
    saliency = saliency[:,:,3]
    desc = generate_description(act)
    exp = generate_explanation(obj_sals)
    sal_img = sa.saliency_analysis_image(masked_s0, obj_sals)

    show_analyze(0, desc, exp, s0, saliency, masked_s0, 0, save=False)

def mask(state, x, y, object, sa):
    obj_tmp =  np.array(sa.tm.templates[object][0])
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    x_len, y_len = obj_tmp.shape[1], obj_tmp.shape[0]
    x_start, y_start = int(x - x_len / 2)+5, int(y - y_len / 2)
    state[y_start: y_start+y_len, x_start: x_start+x_len] = obj_tmp
    return state










def run_submission(cfg, output, nr):
    player = get_player(dumpdir=output)
    predfunc = get_predict_func(cfg)
    for k in range(nr):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)#, task='save_image')
        print("Total:", score)

def do_submit(output, api_key):
    gym.upload(output, api_key=api_key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model', required=True)
    parser.add_argument('--env', help='environment name', required=True)
    parser.add_argument('--episode', help='number of episodes to run',
            type=int, default=100)
    parser.add_argument('--output', help='output directory', default='gym-submit')
    parser.add_argument('--double', help='If use double DQN', default='t')
    parser.add_argument('--dueling', help='If use dueling method', default='f')
    parser.add_argument('--api', help='gym api key')
    #parser.add_argument('--task', help='task to perform', choices=['gym','sample'], default='gym')
    args = parser.parse_args()

    ENV_NAME = args.env
    if args.double == 't':
        DOUBLE = True
    elif args.double == 'f':
        DOUBLE = False
    else:
        logger.error("double argument must be t or f")
    if args.dueling == 't':
        DUELING = True
    elif args.dueling == 'f':
        DUELING = False
    else:
        logger.error("dueling argument must be t or f")

    if DOUBLE:
        logger.info("Using Double")
    if DUELING:
        logger.info("Using Dueling")

    assert ENV_NAME
    logger.info("Environment Name: {}".format(ENV_NAME))
    p = get_player(); del p    # set NUM_ACTIONS

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(args.load),
            input_var_names=['state'],
            output_var_names=['Qvalue'])

    s_cfg = PredictConfig(
            model=Model(),
            session_init=SaverRestore(args.load),
            input_var_names=['state'],
            output_var_names=['saliency'])

    predfunc = OfflinePredictor(cfg)
    s_func   = OfflinePredictor(s_cfg)
    tm = TemplateMatcher('../obj/MsPacman-v0')

    sample_epoch_for_analysis(predfunc, s_func, args.output)
    real_act(tm, args.output)


    exit()
    #analyze('arrays1', args.output)
    #sensitivity_analysis(667, s_cfg, cfg)
    #run_submission(cfg, args.output, args.episode)
    #do_submit(args.output, args.api)
    #for i in xrange(100,300):
    #    object_saliencies(i, cfg, draw=True)
    #object_saliencies(120, cfg, draw=True)

    #acts = [0]
    #for i in xrange(1,1850):
    #    act = object_saliencies(i, predfunc, s_func, tm, draw=True)
    #    acts.append(act)
    #pickle.dump(acts, open('models/DDQN/acts-O-DDQN', 'w'))
    #real_act(tm)
    object_saliencies(641, predfunc, s_func, tm, draw=True)
    #change_points(tm, predfunc)



    #saliency = cv2.resize(saliency, (160, 210))
    #obj_sals = [(-17.05189323425293, 'ghost', Position(left=141, right=151, up=158, down=171))]
    #act = 3

