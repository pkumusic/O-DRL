#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# DL Project for 10807
# Author: Music, Tian, Jing, Yuxin
mkdir -p train_log_copied
mkdir train_log_copied/$2
scp music@shredder:~/O-DRL/DQN/train_log/$2/{log*,latest,graph*,events*,checkpoint} train_log_copied/$2
