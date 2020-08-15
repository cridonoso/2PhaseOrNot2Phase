screen -S 0_lstm -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 0 --dataset ogle --normalization n2 --fold_n 0 --rnn_unit lstm
screen -S 0_phased -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 1 --dataset ogle --normalization n2 --fold_n 0 --rnn_unit plstm

screen -S 1_lstm -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 2 --dataset ogle --normalization n2 --fold_n 1 --rnn_unit lstm
screen -S 1_phased -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 3 --dataset ogle --normalization n2 --fold_n 1 --rnn_unit plstm

screen -S 2_lstm -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 4 --dataset ogle --normalization n2 --fold_n 2 --rnn_unit lstm
screen -S 2_phased -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 5 --dataset ogle --normalization n2 --fold_n 2 --rnn_unit plstm

screen -S 3_lstm -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 6 --dataset ogle --normalization n1 --fold_n 0 --rnn_unit lstm
screen -S 3_phased -X exec ~/anaconda3/envs/tf2/bin/python /home/ubuntu/plstm_tf2/main.py 7 --dataset ogle --normalization n1 --fold_n 1 --rnn_unit lstm
