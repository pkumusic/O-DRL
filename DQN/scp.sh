#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing, Yuxin
mkdir $2
scp music@shredder:~/O-DRL/DQN/train_log/*/{log*,latest,graph*,events*,checkpoint} DQN/train_log_copied
