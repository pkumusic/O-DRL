#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing

# Template directory structure
#
# MsPacman-v0/templates/cherry.png
#                      /dot.png
#             thresholds.pkl
#               ...


# threshoold
# pellet: 0.88
# others: 0.8


# pickle
# pickle.dump(data, outfile=open())
# pickle.load(infile)


import cv2, pickle, pprint, glob, os
import numpy as np
import pylab
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import threading
THRESHOLDS_FILE = 'thresholds.pkl'
TEMPLATES_DIR  = 'templates'

class TemplateMatcher(object):
    def __init__(self, template_dir):
        self.template_dir = template_dir
        self.obj2index, self.index2obj, self.templates = self.read_objects() # Use int index as keys.
        self.thresholds = self.read_thresholds()
        self.lock = threading.Lock()

    def match_all_objects(self, image):
        """ This is the API to extract objects for an image.
            Given an image, return the extracted objects in the image as
            {obj: [(left,right,top,bottom), ..., ]}
        :param image: default as colored image. Height * Width * 3 numpy array
        :return: obj_areas. {obj: [(left,right,top,bottom), ..., ]}
        """
        obj_areas = {}
        for obj in self.obj2index.keys():
            obj_areas[obj] = self.match_object(image, obj)
        return obj_areas

    def match_object(self, image, obj):
        templates  = self.templates[obj]
        thresholds = self.thresholds[obj]
        assert len(templates) == len(thresholds)
        matched_areas = []
        for i in xrange(len(thresholds)):
            matched_template_areas = self.match_template(image, templates[i], thresholds[i])
            # TODO: combine matched_template_areas to create matched_areas of one object. May need to remove duplicates.
            matched_areas.extend(matched_template_areas)
        return matched_areas


    def match_template(self, image, template, threshold, show=False):
        """
        Match the image with one single template. return the matched rectangular areas
        :param image:
        :param template: template file name
        :param threshold:
        :return: [(left,right,top,bottom), (...)]
        """
        object_locs = []
        img_rgb = image
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        self.lock.acquire()
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        self.lock.release()
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            object_locs.append((pt[0], pt[0]+w, pt[1], pt[1] + h))
            if show:
                cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)
        if show:
            plt.imshow(img_rgb)
            plt.show()
        return object_locs


    def fake_match_all_objects(self, image):
        # Used for test the API for train DQN
        self.obj2index = {'pacman':1, 'bean': 2}
        self.index2obj = {1:'pacman', 2:'bean'}
        obj_areas = {1:[(10,20,10,20), (30,40,30,40)], 2:[(1,5,1,5)]}
        return obj_areas

    def read_objects(self):
        """
        :return: objects: {'bean':{0: image, 1: image...}}
        """
        templates_path = os.path.join(self.template_dir, TEMPLATES_DIR)
        objects = defaultdict(dict)
        for filename in glob.glob(templates_path+'/*'):
            template = cv2.imread(filename)
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            object = os.path.basename(filename).split('.')[0]
            infos = object.split('_')
            if len(infos) == 1:
                objects[object][0] = template
            elif len(infos) == 2:
                object, index = infos[0], int(infos[1])
                objects[object][index] = template
            else:
                print "ERROR: template name should not include _ character."
                exit()
        object_index, obj2index, index2obj = 0, {}, {}
        for object in objects.keys():
            obj2index[object] = object_index
            index2obj[object_index] = object
            object_index += 1
        #pprint.pprint(obj2index)

        print "objects", index2obj
        return obj2index, index2obj, objects

    def read_thresholds(self):
        #TODO: fine tune thresholds and save it to file
        return             {'ghost': [0.7, 0.7],
                           'pacman': [0.8, 0.8, 0.8, 0.8, 0.8, 0.8],
                           'cherry': [0.8],
                           'dot': [0.8],
                            'pellet': [0.88],
                           'eatable': [0.7]}
        # thresholds_path = os.path.join(self.template_dir, THRESHOLDS_FILE)
        # pkl_file = open(thresholds_path, 'r')
        # thresholds = pickle.load(pkl_file)
        # pkl_file.close()
        # return thresholds

    def draw_extracted_image(self, image, extracted_objects):
        for object, locs in extracted_objects.iteritems():
            for loc in locs:
                cv2.rectangle(image, (loc[0],loc[2]), (loc[1], loc[3]), (0, 0, 255), 1)
                cv2.putText(image, object, (loc[0]-2, loc[2]), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)

        plt.imshow(image)
        plt.show()


    def process_image(self, image, obj_areas, method='swap_input_combine', debug=False):
        if method.split('_')[2] == 'combine':
            image = np.zeros((image.shape[:2]))
            for obj, areas in obj_areas.iteritems():
                for area in areas:
                    image[area[2]:area[3]+1,area[0]:area[1]+1] = self.obj2index[obj]+1
            return image
        if method.split('_')[2] == 'separate':
            obj_images = np.zeros((image.shape[0], image.shape[1], len(self.obj2index)))
            for obj, areas in obj_areas.iteritems():
                obj_index = self.obj2index[obj]
                for area in areas:
                    obj_images[area[2]:area[3]+1, area[0]:area[1]+1, obj_index] = 1
            return obj_images






if __name__ == '__main__':
    tm = TemplateMatcher('../obj/MsPacman-v0')
    # Test on single template
    #template = cv2.imread('../obj/MsPacman-v0/templates/eatable.png')
    #tm.match_template(image, template, 0.7, show=True)

    # Test on all objects
    for i in xrange(100, 200):
        i = np.random.randint(1, 1229)
        image = np.load('../obj/MsPacman-v0-sample/%s.npy'%i)
        extracted_objects = tm.match_all_objects(image)
        tm.draw_extracted_image(image, extracted_objects)



    # tm.match_template('test.png', '../obj/templates/ghost_3.png')
    # tm.match_template('test.png', '../obj/templates/ghost_1.png')
    # tm.match_template('test.png', '../obj/templates/ghost_1.png')
    # tm.match_template('test.png', '../obj/templates/pacman.png')
    # tm.match_template('test.png', '../obj/templates/pellet.png', threshold=0.88)
    # tm.match_template('test.png', 'template.png')

def generate_templates():
    img_array = np.load('../obj/MsPacman-v0-sample/605.npy')
    # print img_array.shape

    # use matplot to show image
    plt.imshow(img_array)
    plt.show()

    plt.imsave('test.png', img_array)

    # extract template
    # img = img_array[80:92, 75:85]  ## extract ghost_left/right with 1.npy
    # img = img_array[26:38, 95:105] ## extract ghost change with 600.npy
    # img = img_array[102:106, 7:13]  ## extract dot with 600.npy
    # img = img_array[145:154, 147:153]  ## extract pellets with 600.npy
    # img = img_array[74:86, 28:38] ## extract pacman with 600.npy
    # img = img_array[50:62, 78:89] ## extract cherry with 605.npy
    # cv2.imwrite('template.png', img)

# generate_templates()

def thresholds():
    out = {'ghost':[0.8, 0.8], 'pacman':[0.8], 'cherry':[0.8], 'dot':[0.8], 'pellet':[0.88], 'eatable':[0.8]}
    outfile = open('../obj/thresholds.pkl', 'w')
    pickle.dump(out, outfile)
    outfile.close()

    pkl_file = open('../obj/thresholds.pkl', 'r')
    data = pickle.load(pkl_file)
    pprint.pprint(data)
    pkl_file.close()

#thresholds()
