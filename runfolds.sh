screen -S 0_lstm -X exec ~/anaconda3/envs/tf2/bin/python main.py 0 --dataset wise --normalization n1 --fold_n 0 --rnn_unit lstm
screen -S 0_phased -X exec ~/anaconda3/envs/tf2/bin/python main.py 1 --dataset wise --normalization n1 --fold_n 0 --rnn_unit plstm

screen -S 1_lstm -X exec ~/anaconda3/envs/tf2/bin/python main.py 2 --dataset wise --normalization n1 --fold_n 1 --rnn_unit lstm
screen -S 1_phased -X exec ~/anaconda3/envs/tf2/bin/python main.py 3 --dataset wise --normalization n1 --fold_n 1 --rnn_unit plstm

screen -S 2_lstm -X exec ~/anaconda3/envs/tf2/bin/python main.py 4 --dataset wise --normalization n1 --fold_n 2 --rnn_unit lstm
screen -S 2_phased -X exec ~/anaconda3/envs/tf2/bin/python main.py 5 --dataset wise --normalization n1 --fold_n 2 --rnn_unit plstm

