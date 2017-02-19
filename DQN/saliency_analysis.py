#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music

# Analyze the produced saliency maps
# Read states and saliency maps from dir and process with object recognize
from __future__ import division
from obj_recognizor import TemplateMatcher
import numpy as np
import os
import cv2

def mean(l):
    return sum(l) / len(l)


class Saliency_Analyzor():
    def __init__(self, template_dir):
        self.tm = TemplateMatcher(template_dir)

    def object_saliencies(self, state, saliency):
        """
        :param state: The state matrix
        :param saliency: The saliency matrix
        :return: a list of object saliencies: [(saliency, obj, Position), ..., ]
        """
        assert saliency.shape == (210, 160)
        tm = self.tm
        extracted_objects = tm.match_all_objects(state)
        #tm.draw_extracted_image(state, extracted_objects)
        obj_sals = []
        for obj, locs in extracted_objects.iteritems():
            for loc in locs:
                sal =  self.calc_obj_sal(saliency, loc)
                obj_sals.append((sal, obj, loc))
        return obj_sals

    def object_saliencies_filter(self, obj_sals, topk=3):
        pos_obj_sals = sorted([sal for sal in obj_sals if sal[0] > 0], reverse=True)
        neg_obj_sals = sorted([sal for sal in obj_sals if sal[0] < 0])
        return pos_obj_sals[:2], neg_obj_sals[:2]

    def calc_obj_sal(self, saliency, loc, gamma=0.1):
        """
        :param saliency: The saliency matrix
        :param loc: The location of the object bounding box
        :return: The value of the object saliency
        """
        sals = saliency[loc.up:loc.down, loc.left:loc.right]
        sals = list(sals.flatten())
        pos_sals = sorted([sal for sal in sals if sal > 0], reverse=True)
        neg_sals = sorted([sal for sal in sals if sal < 0])
        if len(pos_sals) > len(neg_sals):
            return mean(pos_sals[:1+int(gamma*len(pos_sals))])
        else:
            return mean(neg_sals[:1+int(gamma*len(pos_sals))])


    def saliency_image(self, image, pos_obj_sals, neg_obj_sals, filePath=None):
        import matplotlib.pyplot as plt
        assert image.shape == (210, 160, 3)
        for obj_sal in pos_obj_sals:
            (sal, obj, loc) = obj_sal
            image = image.copy() # A bug in opencv
            cv2.rectangle(image, (loc.left, loc.up), (loc.right, loc.down), (0, 0, 255), 1)
            cv2.putText(image, "%s:%.2f" %(obj,sal), (loc.left - 2, loc.up), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1,
                        cv2.LINE_AA)
        for obj_sal in neg_obj_sals:
            (sal, obj, loc) = obj_sal
            image = image.copy() # A bug in opencv
            cv2.rectangle(image, (loc.left, loc.up), (loc.right, loc.down), (255, 0, 0), 1)
            cv2.putText(image, "%s:%.2f" %(obj,sal), (loc.left - 2, loc.up), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1,
                        cv2.LINE_AA)
        #show_image(image)
        return image

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
    obj_sals = sa.object_saliencies(state, saliency)
    obj_sals = sa.object_saliencies_filter(obj_sals)
    image = sa.saliency_image(state, obj_sals)
    show_image(image)
    show_image(saliency)



