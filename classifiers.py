import tensorflow as tf
from phased import PhasedLSTM
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import LSTM, RNN, LayerNormalization
import numpy as np
import time


def mask_pred(y_pred, mask):
    last = tf.cast(tf.reduce_sum(mask, 1), tf.int32)-1
    last = tf.stack([tf.range(last.shape[0]), last], 1)
    last_probas  = tf.gather_nd(y_pred, last)
    last_probas  = tf.expand_dims(last_probas, 2)
    tensor_last  = tf.tile(last_probas, [1,1,y_pred.shape[1]])
    inverse_mask = tf.cast(tf.where(mask == 0, 1, 0), dtype=tf.float32)
    tensor_last  = tf.transpose(tensor_last, [0,2,1]) * tf.expand_dims(inverse_mask, 2)
    tensor_first = y_pred * tf.expand_dims(mask, 2)
    return tensor_first + tensor_last

# =====================================================
# ================= PHASED CLASSIFIER =================
# =====================================================
class PhasedClassifier(tf.keras.Model):

    def __init__(self, units, n_classes, dropout=0.5):
        super(PhasedClassifier, self).__init__()
        self._cells = []
        self._units = units

        self.plstm_0 = PhasedLSTM(self._units, name='rnn_0')
        self.plstm_1 = PhasedLSTM(self._units, name='rnn_1')
        self.fc = tf.keras.layers.Dense(n_classes,
                                        activation='softmax',
                                        dtype='float32')
        self.dropout = tf.keras.layers.Dropout(dropout)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    @tf.function
    def call(self, inputs, times, training=False):
        states_0 = (tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units]))

        states_1 = (tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units]))

        initial_state = (states_0, states_1)


        x_t = tf.transpose(inputs, [1, 0, 2])
        t_t = tf.transpose(times, [1, 0])
        
        time_steps = tf.shape(x_t)[0]

        def compute(i, cur_state, out):
            output_0, cur_state0 = self.plstm_0(x_t[i], t_t[i], cur_state[0])
            output_1, cur_state1 = self.plstm_1(output_0, t_t[i], cur_state[1])
            output_2 = self.dropout(output_1)
            return tf.add(i, 1), (cur_state0, cur_state1), out.write(i, self.fc(output_2))

        # start_time = time.time()
        _, cur_state, out = tf.while_loop(
            lambda a, b, c: a < time_steps,
            compute,
            (tf.constant(0), initial_state, tf.TensorArray(tf.float32, time_steps))
        )
        # end_time = time.time()
        # tf.print('takes: {} sec'.format(end_time-start_time))

        return tf.transpose(out.stack(), [1,0,2])

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
                    y_pred = self(train_batch[0], train_batch[0][...,0], training=True)
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

    def predict_proba(self, test_batches):
        t0 = time.time()
        test_losses = []
        predictions = []
        true_labels = []
        for test_batch in test_batches:
            y_pred = self(test_batch[0], test_batch[0][...,0])
            loss_value = self.get_loss(test_batch[1], y_pred, test_batch[2])
            test_losses.append(loss_value)
            predictions.append(mask_pred(y_pred, test_batch[2]))
            true_labels.append(test_batch[1])
        avg_epoch_loss_val = tf.reduce_mean(test_losses).numpy()


        t1 = time.time()
        print('runtime {:.2f}'.format((t1-t0)))
        predictions = tf.concat(predictions, 0)
        true_labels = tf.concat(true_labels, 0)
        return predictions, true_labels, avg_epoch_loss_val


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


# =====================================================
# ================= LSTM CLASSIFIER =================
# =====================================================
class LSTMClassifier(tf.keras.Model):

    def __init__(self, units, n_classes,dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self._units = units

        self.lstm_0 = LSTM(self._units, dropout=dropout, return_sequences=True, name='rnn_0')
        self.lstm_1 = LSTM(self._units, dropout=dropout, return_sequences=True, name='rnn_1')
        
        self.fc = tf.keras.layers.Dense(n_classes,
                                        activation='softmax',
                                        dtype='float32')

        self.norm_layer = LayerNormalization()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    @tf.function
    def call(self, inputs, masks, training=False):
        state_0 = (tf.zeros([inputs.shape[0], self._units]),
                   tf.zeros([inputs.shape[0], self._units]))

        state_1 = (tf.zeros([inputs.shape[0], self._units]),
                   tf.zeros([inputs.shape[0], self._units]))

        # start_time = time.time()

        masks = tf.cast(masks, tf.bool)

        output_0 = self.lstm_0(inputs, mask=masks, training=training,initial_state=state_0)
        output_0 = self.norm_layer(output_0)
        output_1 = self.lstm_1(output_0, mask=masks, training=training,initial_state=state_1)
        output_1 = self.norm_layer(output_1)
        out = self.fc(output_1)

        # end_time = time.time()
        # tf.print('takes: {} sec'.format(end_time-start_time))

        return out

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
                    y_pred = self(train_batch[0], train_batch[2], training=True)
                    loss_value = self.get_loss(train_batch[1], y_pred, train_batch[2])

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            print('validation')

            val_losses = []
            for val_batch in val:
                y_pred = self(val_batch[0], val_batch[2])
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

    def predict_proba(self, test_batches):
        t0 = time.time()
        test_losses = []
        predictions = []
        true_labels = []

        for test_batch in test_batches:
            y_pred = self(test_batch[0], test_batch[2])
            loss_value = self.get_loss(test_batch[1], y_pred, test_batch[2])
            test_losses.append(loss_value)
            predictions.append(y_pred)
            true_labels.append(test_batch[1])
        avg_epoch_loss_val = tf.reduce_mean(test_losses).numpy()

        t1 = time.time()
        print('runtime {:.2f}'.format((t1-t0)))
        predictions = tf.concat(predictions, 0)
        true_labels = tf.concat(true_labels, 0)
        return predictions, true_labels, avg_epoch_loss_val


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