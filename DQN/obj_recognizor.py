#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing

# Template directory structure
#
# MsPacman-v0/pacman:  {'templates': [image1, image2 ...], 'thresholds': [float, float, ...]}
#             bean:
#               ...
import numpy as np
import pylab
import matplotlib.pyplot as plt

class TemplateMatcher(object):
    def __init__(self, template_dir):
        self.template_dir = template_dir
        self.obj2index = {}  # e.g., {'pacman':1; 'bean': 2; ...}
        self.index2obj = {}
        self.obj_dict = self.read_objects() # Use int index as keys.

    def match_all_objects(self, image):
        """ This is the API to extract objects for an image.
            Given an image, return the extracted objects in the image as
            {obj: [(left,right,top,bottom), ..., ]}
        :param image: default as colored image. Height * Width * 3 numpy array
        :return: obj_areas. {obj: [(left,right,top,bottom), ..., ]}
        """
        obj_areas = {}
        for obj in self.obj_dict:
            obj_areas[obj] = self.match_object(image, obj)
        return obj_areas


    def match_object(self, image, obj):
        templates  = self.obj_dict[obj]['templates']
        thresholds = self.obj_dict[obj]['thresholds']
        assert len(templates) == len(thresholds)
        matched_areas = []
        for i in xrange(len(templates)):
            matched_template_areas = self.match_template(image, templates[i], thresholds[i])
        #TODO: combine matched_template_areas to create matched_areas of one object. May need to remove duplicates.
        return matched_areas

    def match_template(self, image, template, threshold):
        """
        Match the image with one single template. return the matched rectangular areas
        :param image:
        :param template:
        :param threshold:
        :return: [(left,right,top,bottom), (...)]
        """
        pass

    def read_objects(self):
        """
        Read obj files in the directory and create a dictionary contain the mapping from obj -> templates
        Initialize obj2index and index2obj.
        :return: {'obj1', templates}; where templates is a dictionary
                                    like {'templates': [image1, image2 ...], 'thresholds': [float, float, ...]}
        """
        pass

    def fake_match_all_objects(self, image):
        # Used for test the API for train DQN
        self.obj2index = {'pacman':1, 'bean': 2}
        self.index2obj = {1:'pacman', 2:'bean'}
        obj_areas = {1:[(10,20,10,20), (30,40,30,40)], 2:[(1,5,1,5)]}
        return obj_areas

    @staticmethod
    def process_image(image, obj_areas, method='swap_input_combine'):
        if method == 'swap_input_combine':

            plt.imshow(image)
            plt.show()
            image = np.zeros((image.shape[:2]))
            for obj, areas in obj_areas.iteritems():
                for area in areas:
                    image[area[0]:area[1]+1,area[2]:area[3]+1] = obj
            plt.imshow(image, cmap=pylab.gray())
            plt.show()
            exit()
            return image


if __name__ == '__main__':
    tm = TemplateMatcher('../obj/MsPacman-v0')

