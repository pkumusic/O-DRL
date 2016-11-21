#!/usr/bin/env bash
scp music@shredder:~/O-DRL/DQN/train_log/*/{log*,latest,graph*,events*,checkpoint} DQN/train_log_copied
