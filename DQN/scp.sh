mkdir $2
scp music@shredder:~/tensorpack_music/examples/$1/train_log/$2/log* music@shredder:~/tensorpack_music/examples/$1/train_log/$2/latest music@shredder:~/tensorpack_music/examples/$1/train_log/$2/graph* music@shredder:~/tensorpack_music/examples/$1/train_log/$2/events* music@shredder:~/tensorpack_music/examples/$1/train_log/$2/checkpoint $2/
