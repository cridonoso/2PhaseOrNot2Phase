import tensorflow as tf
from phased import PhasedLSTM
from tensorflow.keras.losses import categorical_crossentropy
import numpy as np
import time


class RNNClassifier(tf.keras.Model):

    def __init__(self, units):
        super(RNNClassifier, self).__init__()
        self._cells = []
        self._units = units

        self.plstm_0 = PhasedLSTM(self._units, name='rnn_0')
        self.plstm_1 = PhasedLSTM(self._units, name='rnn_1')
        self.fc = tf.keras.layers.Dense(5,
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
    def get_loss(self, y_true, y_pred, masks):
        y_expanded = tf.tile(tf.expand_dims(y_true, 1), 
                             [1, y_pred.shape[1], 1])
        loss = categorical_crossentropy(y_expanded, y_pred)
        masked_loss = loss * masks
        cumulated_loss = tf.reduce_sum(masked_loss, 1)
        return tf.reduce_mean(cumulated_loss)

    def fit(self, train, val, epochs, patience=5, save_path='.'):

        # Define checkpoints
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                   model=self,
                                   optimizer=self.optimizer)

        manager = tf.train.CheckpointManager(ckpt,
                                             save_path+'/phased_{}'.format(self._units)+'/ckpts',
                                             max_to_keep=1)
        # Training variables
        best_loss = 9999999
        early_stop_count = 0

        for epoch in range(epochs):
            t0 = time.time()
            for train_batch in train:
                with tf.GradientTape() as tape:
                    y_pred = self(train_batch[0], train_batch[0][...,0])
                    loss_value = self.get_loss(train_batch[1], y_pred, train_batch[2])

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            print('validation')

            val_losses = []
            for val_batch in val:
                y_pred = self(val_batch[0], val_batch[0][...,0])
                loss_value = self.get_loss(val_batch[1], y_pred, val_batch[2])
                val_losses.append(loss_value)

            t1 = time.time()
            avg_epoch_loss_val = tf.reduce_mean(val_losses).numpy()
            print('({:.2f} sec) Epoch: {} -  val loss: {:.2f}'.format((t1-t0), epoch, avg_epoch_loss_val))

            # EARLY STOPPING
            if  avg_epoch_loss_val < best_loss:
                best_loss = avg_epoch_loss_val
                manager.save()
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count == patience:
                print('EARLY STOPPING ACTIVATED')
                break

    def load_ckpt(self, ckpt_path):
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                   model=self,
                                   optimizer=self.optimizer)

        manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
        ckpt = ckpt.restore(manager.latest_checkpoint).expect_partial()
        if not manager.latest_checkpoint:
            print("Initializing from scratch.")
        else:
            print('RNN Restored!')