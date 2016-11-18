#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing
# This file is used to sample images for pre-training object recognition
# Building template and its weights.

from tensorpack.RL.expreplay import ExpReplay
from tensorpack.RL.gymenv import GymEnv
from tensorpack.RL.common import MapPlayerState, PreventStuckPlayer, LimitLengthPlayer
from tensorpack.RL.history import HistoryFramePlayer
import cv2
import numpy as np
from tensorpack.utils.utils import get_rng
ENV_NAME = 'MsPacman-v0'
IMAGE_SIZE = (84, 84)
FRAME_HISTORY = 4

def get_player(viz=False, train=False, dumpdir=None):
    pl = GymEnv(ENV_NAME, dumpdir=dumpdir)
    #TODO: Preprocessing goes here
    def func(img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = img[:,:,0]
        #img = cv2.resize(img, IMAGE_SIZE[::])
        return img
    pl = MapPlayerState(pl, func)

    global NUM_ACTIONS
    NUM_ACTIONS = pl.get_action_space().num_actions()
    if not train:
        pass
        #pl = HistoryFramePlayer(pl, FRAME_HISTORY)
        #pl = PreventStuckPlayer(pl, 30, 1)
    pl = LimitLengthPlayer(pl, 40000)
    return pl

def detect(image, obj, method='absolute', max_diff=0.25):
    """ A naive method to find the position of the template object in the image
    :param image:
    :param obj:
    :return:
    """
    x, y = image.shape
    print 'object shape', obj.shape
    xo, yo = obj.shape
    #TODO: Reduce time complexity to O(x*y) by tracking the change.
    answers = []
    for i in xrange(x):
        for j in xrange(y):
            im = image[i:i+xo, j:j+yo]
            if im.shape == obj.shape:
                if method == 'absolute':
                    diff = float(np.sum(im != obj)) / obj.size
                elif method == 'relative':
                    diff = np.sum(abs(im - obj) / 255.0) / obj.size
            if diff < max_diff:
                answers.append((i, j))
    return answers

if __name__ == '__main__':
    player = get_player()
    rng = get_rng()

    task = 'detect'

    import matplotlib.pyplot as plt
    if task == 'save':
        for i in xrange(50):
            #random_action = rng.choice(range(NUM_ACTIONS))
            player.action(1)
            # Original image: (210 * 160 * 3) by print player.current_state().shape
            # print player.current_state().shape
            #if i == 25:
            #    player.restart_episode()
            file_name = 'freeway/' + ENV_NAME + '_' + str(i)
            X =  player.current_state()
            np.save(file_name, X)
            imgplot = plt.imshow(X)
            plt.savefig(file_name)

    # Experiment on picture 20
    # MANUALLY EXTRACT OBJECTS TEMPLATES
    if task == 'template':
        sample = '20'
        file_name = 'freeway/' + ENV_NAME + '_' + str(sample)
        image = np.load(file_name + '.npy')
        #Canny Edge Detection
        # plt.imshow(image)
        # plt.show()
        #image = cv2.Canny(image, 30, 100, 5)
        #image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        #image = cv2.Laplacian(image,cv2.CV_64F)
        # plt.imshow(image)
        # plt.show()
        # exit()

        #Find the template
        # For
        # Y = image[27:37, 143:151] #o1
        # Y = image[43:53, 139:147] #o2
        # Y = image[59:69, 134:142] #o3
        # Y = image[75:85, 122:130] #o4
        # Y = image[91:101, 88:96]  #o5
        # Y = image[107:117, 66:74] #o6
        # Y = image[123:133, 32:40] #o7
        # Y = image[139:149, 20:28] #o8
        # Y = image[155:165, 15:23] #o9
        # Y = image[171:181, 11:19] #o10
        # Y = image[125:133,44:50] #o11
        #np.save('freeway/o11.npy', Y)
        o = Y
        #o = np.load('samples/o1.npy')
        xo,yo = o.shape

        # print o
        plt.subplot(1, 2, 1)
        plt.imshow(o)
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.show()
        # indices = detect(image, o, max_diff=0.1)
        # print indices
        # for (i, j) in indices:
        #     im = image[i:i+xo, j:j+yo]
        #     plt.imshow(im)
        #     plt.show()

        #plt.imshow(X)
        # print o1
        #plt.show()
    if task == 'detect':
        sample = '24'
        file_name = 'freeway/' + ENV_NAME + '_' + str(sample)
        image = np.load(file_name + '.npy')

        objs = [str(i) for i in xrange(1, 12)]
        for id in objs:
            obj = np.load('freeway/' + 'o' + id + '.npy')
            indices = detect(image, obj)
            for index in indices:
                x, y = index
                cv2.rectangle(image, (y,x), (y+obj.shape[0], x+obj.shape[1]), (0,255,0), 1)
        plt.imshow(image)
        plt.show()