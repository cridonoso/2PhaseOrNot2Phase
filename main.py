from phased import PhasedLSTM
import tensorflow as tf
import time


units = 128
input_dim = 3
n_samples = 32
time_steps = 1000
plstm = PhasedLSTM(units)
times = tf.random.truncated_normal([n_samples, time_steps])
observations = tf.random.truncated_normal([n_samples, time_steps, input_dim])

states = (tf.zeros([observations.shape[0], units]),
          tf.zeros([observations.shape[0], units]))

@tf.function
def run_phased(observations, times, states):
    for step in tf.range(time_steps):
        output, states = plstm(observations[:,step,:], times[:,step], states)


t0 = time.time()
run_phased(observations, times, states)
t1 = time.time()
print('total time : {}'.format(t1-t0))
