#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music

# Analyze the produced saliency maps
# Read states and saliency maps from dir and process with object recognize

from obj_recognizor import TemplateMatcher
import numpy as np
import os

class Saliency_Analyzor():
    def __init__(self, template_dir):
        self.tm = TemplateMatcher(template_dir)

    def object_saliencies(self, state, saliency):
        """
        :param state: The state matrix
        :param saliency: The saliency matrix
        :return: a list of object saliencies: [(saliency, obj, Position), ..., ]
        """
        tm = self.tm
        extracted_objects = tm.match_all_objects(state)
        tm.draw_extracted_image(state, extracted_objects)
        print extracted_objects

def read_state_and_saliency(dir, index):
    state = np.load(os.path.join(dir, 'state%d.npy' %index))
    saliency = np.load(os.path.join(dir, 'saliency%d.npy' %index))
    return state, saliency

def show_image(img):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()



if __name__ == "__main__":
    sa = Saliency_Analyzor('../obj/MsPacman-v0')
    state, saliency = read_state_and_saliency('sal-original', 1)
    sa.object_saliencies(state, saliency)



