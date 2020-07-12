screen -S 0_lstm -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 0 --dataset $1 --normalization $2 --fold_n 0 --rnn_unit lstm
screen -S 0_phased -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 1 --dataset $1 --normalization $2 --fold_n 0 --rnn_unit plstm

screen -S 1_lstm -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 2 --dataset $1 --normalization $2 --fold_n 1 --rnn_unit lstm
screen -S 1_phased -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 3 --dataset $1 --normalization $2 --fold_n 1 --rnn_unit plstm

screen -S 2_lstm -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 4 --dataset $1 --normalization $2 --fold_n 2 --rnn_unit lstm
screen -S 2_phased -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 5 --dataset $1 --normalization $2 --fold_n 2 --rnn_unit plstm

