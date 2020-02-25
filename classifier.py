import tensorflow as tf
from phased import PhasedLSTM
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import time


class RNNClassifier(tf.keras.Model):

    def __init__(self, units, n_classes):
        super(RNNClassifier, self).__init__()
        self._cells = []
        self._units = units
        self.n_classes = n_classes
        self.plstm_0 = PhasedLSTM(self._units, name='rnn_0')
        self.plstm_1 = PhasedLSTM(self._units, name='rnn_1')
        self.fc = tf.keras.layers.Dense(self.n_classes,
                                        activation='softmax',
                                        dtype='float32')
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def call(self, inputs, times):
        states_0 = (tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units]))

        states_1 = (tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units]))

        predictions = tf.TensorArray(tf.float32, size=inputs.shape[1])

        for step in tf.range(inputs.shape[1]):
            output_0, states_0 = self.plstm_0(inputs[:,step,:], times[:,step], states_0)
            output_1, states_1 = self.plstm_1(output_0, times[:,step], states_1)
            y_pred = self.fc(output_1)
            predictions = predictions.write(step, y_pred)

        return tf.transpose(predictions.stack(), [1,0,2])

    @tf.function
    def get_loss(self, y_true, y_pred, lengths):
        y_one_hot = tf.one_hot(y_true, self.n_classes)
        sum_losses = tf.zeros(lengths.shape[0])
        for step in tf.range(lengths.shape[1]):
            loss = categorical_crossentropy(y_one_hot, y_pred[:, step, :])
            value = loss * lengths[:,step]
            sum_losses+=value
        return tf.reduce_mean(sum_losses)


    def fit(self, train, val, epochs, patience=5):
        t0 = time.time()
        for epoch in range(epochs):
            for batch in train:
                with tf.GradientTape() as tape:
                    y_pred = self(batch[0], batch[1])
                    loss_value = self.get_loss(batch[2], y_pred, batch[3])
                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            print('validation')
            val_losses = []
            for val_batch in val:
                y_pred = self(val_batch[0], val_batch[1])
                loss_value = self.get_loss(val_batch[2], y_pred, val_batch[3])
                val_losses.append(loss_value)
            t1 = time.time()
            print('({} sec) {} -  val loss: {}'.format((t1-t0), epoch, np.mean(val_losses)))
