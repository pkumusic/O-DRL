#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music

# Used for generating videos for human subjects
import argparse
import numpy as np
import pickle

def sample_data(index, input, draw=False):
    arrays = np.load(input + '/arrays%d.npz' % index)
    state, pixel, object = arrays['s0'], arrays['pixel_saliency'], arrays['object_saliency']
    real_acts = pickle.load(input + '/real_acts')
    real_act = real_acts[index]
    print "real_action", real_act
    print state
    print pixel
    print object
    if draw:
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.axis('off')
        fig = plt.imshow(state, aspect='equal')
        plt.title(real_act)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # Object Saliency
        plt.subplot(132)
        plt.axis('off')
        fig = plt.imshow(object, cmap='gray', aspect='equal')
        plt.title('Object Saliency Map')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # colorbar = Colorbar(fig, location='upper left')
        # colorbar.set_ticks([-100,-50,0,50,100])
        # plt.gca().add_artist(colorbar)
        # Pixel Saliency
        plt.subplot(133)
        plt.axis('off')
        fig = plt.imshow(pixel, cmap='gray', aspect='equal')
        plt.title('Pixel Saliency Map')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        # colorbar = Colorbar(fig, location='upper left')
        # colorbar.set_ticks([-0.03,0,0.03])
        # plt.gca().add_artist(colorbar)
        plt.savefig('test/' + str(index), bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input directory', required=True)
    args = parser.parse_args()
    sample_data(1, args.input,draw=True)