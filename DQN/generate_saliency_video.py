#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Music

# Used for generating videos for human subjects
import argparse
import numpy as np

def sample_data(index, input):
    arrays = np.load(input + '/arrays%d.npz' % index)
    state, pixel, object = arrays['s0'], arrays['pixel_saliency'], arrays['object_saliency']
    print state
    print pixel
    print object

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input directory', required=True)
    args = parser.parse_args()
    sample_data(1, args.input)