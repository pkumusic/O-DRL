echo "Usage: sh tensorboard.sh folder method Game port"
echo "Example sh tensorboard.sh DQN DDQN MsPacman 1111"
tensorboard --logdir $1/train_log/$2-$3 --port $4
