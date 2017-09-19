train-DQN:
	python DQN-gym-train.py -g 0 -e MsPacman-v0 --double f --dueling t --logdir dueling-MsPacman-v0

train-O-DQN-4:
	python O-DQN-train.py -g 0 -e MsPacman-v0 --double f --dueling t --logdir O-dueling-4-MsPacman-v0 --object swap_input_separate

train-O-DQN-3:
	python O-DQN-train.py -g 0 -e MsPacman-v0 --double f --dueling t --logdir O-dueling-3-MsPacman-v0 --object add_input_separate

train-O-DQN-2:
	python O-DQN-train.py -g 0 -e MsPacman-v0 --double f --dueling t --logdir O-dueling-2-MsPacman-v0 --object add_input_combine

train-O-DQN-1:
	python O-DQN-train.py -g 0 -e MsPacman-v0 --double f --dueling t --logdir O-dueling-1-MsPacman-v0 --object swap_input_combine

saliency-DDQN:
	python DQN-gym-run-saliency.py --load models/DDQN/model-10m --env MsPacman-v0 --output test --dueling t --double f --episode 1

saliency-O-DDQN:
	python DQN-gym-run-saliency.py --load models/O-DDQN/model-10m --env MsPacman-v0 --output test --dueling f --double t --episode 1

saliency-DQN:
	python DQN-gym-run-saliency.py --load models/O-DDQN/DQN.model-25k --env MsPacman-v0 --output test --dueling f --double f --episode 1

saliency-DDDQN:
	python DQN-gym-run-saliency.py --load models/O-DDQN/DDDQN.model-30k --env MsPacman-v0 --output test --dueling t --double t --episode 1

saliency-O-DDQN-time:
	python DQN-gym-run-saliency.py --load models/DDQN/model-10m --env MsPacman-v0 --output test --dueling t --double f --episode 1

saliency:
	python DQN-gym-run-saliency.py --load dueling.model --env MsPacman-v0 --output arrays2 --dueling t --double f --episode 1

saliency-start:
	python DQN-gym-run-saliency.py --load dueling.model.100000 --env MsPacman-v0 --output arrays2 --dueling t --double f --episode 1

video:
	ffmpeg -r 5 -f image2 -i /Users/MusicLee/Desktop/Coding/O-DRL/DQN/test/file%04d.png -vcodec mpeg4 saliency.avi



