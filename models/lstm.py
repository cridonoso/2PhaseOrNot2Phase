import tensorflow as tf
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import LayerNormalization, LSTMCell, RNN
import time

from os import path
from .tools import mask_pred, add_scalar_log

class LSTMClassifier(tf.keras.Model):

    def __init__(self, units, n_classes, dropout=0.5, lr=1e-3, name='phased'):
        super(LSTMClassifier, self).__init__()
        self._units = units
        self._name  = name

        self.lstm_0 = LSTMCell(self._units, name='rnn_0')
        self.lstm_1 = LSTMCell(self._units, name='rnn_1')

        self.fc = tf.keras.layers.Dense(n_classes,
                                        activation='softmax',
                                        dtype='float32')

        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm_layer = LayerNormalization()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def call(self, inputs, training=False):
        states_0 = [tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units])]

        states_1 = [tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units])]

        initial_state = (states_0, states_1)


        x_t = tf.transpose(inputs, [1, 0, 2])

        time_steps = tf.shape(x_t)[0]

        def compute(i, cur_state, out):
            output_0, cur_state0 = self.lstm_0(x_t[i], cur_state[0])
            output_0 = self.norm_layer(output_0)
            output_1, cur_state1 = self.lstm_1(output_0, cur_state[1])
            output_2 = self.dropout(output_1)
            return tf.add(i, 1), (cur_state0, cur_state1), out.write(i, self.fc(output_2))

        _, cur_state, out = tf.while_loop(
            lambda a, b, c: a < time_steps,
            compute,
            (tf.constant(0), initial_state, tf.TensorArray(tf.float32, time_steps))
        )

        y = out.stack()
        return tf.transpose(y, [1,0,2])


    @tf.function
    def get_loss(self, y_true, y_pred, masks):
        y_expanded = tf.tile(tf.expand_dims(y_true, 1),
                             [1, y_pred.shape[1], 1])

        loss = categorical_crossentropy(y_expanded, y_pred)
        masked_loss = loss * masks
        cumulated_loss = tf.reduce_sum(masked_loss, 1)
        return tf.reduce_mean(cumulated_loss)

    @tf.function
    def get_acc(self, y_true, y_pred, masks):
        last = tf.cast(tf.reduce_sum(masks, 1), tf.int32)-1
        last = tf.stack([tf.range(last.shape[0]), last], 1)
        last_probas  = tf.gather_nd(y_pred, last)
        acc = tf.keras.metrics.categorical_accuracy(y_true, last_probas)
        return tf.reduce_mean(acc)

    def fit(self, train, val, epochs, patience=5, save_path='.'):

        # Tensorboard
        train_log_dir = '{}/{}/logs/train'.format(save_path, self._name)
        test_log_dir  = '{}/{}/logs/val'.format(save_path, self._name)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        ckpts_path = save_path+'/{}'.format(self._name)+'/ckpts'

        # Define checkpoints
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                   model=self,
                                   optimizer=self.optimizer)

        manager = tf.train.CheckpointManager(ckpt,
                                             ckpts_path,
                                             max_to_keep=1)


        if path.isdir(ckpts_path):
            self.load_ckpt(ckpts_path)
            print('[INFO] WEIGHTS SUCCEFULLY LOADED')


        # Training variables
        best_loss = 9999999
        early_stop_count = 0
        iter_count = 0 # each forward pass is an iteration
        curr_eta = 0.
        for epoch in range(epochs):
            print('[INFO] RUNNING EPOCH {}/{} - EARLY STOP: {}/{}'.format(epoch, epochs, early_stop_count, patience), end='\r')

            # =================================
            # ========= TRAINING STEP =========
            # =================================
            for train_batch in train:
                with tf.GradientTape() as tape:
                    y_pred = self(train_batch[0], training=True)

                    loss_value = self.get_loss(train_batch[1], y_pred, train_batch[2])
                    acc_value = self.get_acc(train_batch[1], y_pred, train_batch[2])

                add_scalar_log(acc_value, train_summary_writer, iter_count, 'accuracy')
                add_scalar_log(loss_value, train_summary_writer, iter_count, 'loss')

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                iter_count+=1

            # =================================
            # ======== VALIDATION STEP ========
            # =================================
            val_losses = []
            val_accura = []
            for val_batch in val:
                y_pred = self(val_batch[0], training=False)
                loss_value = self.get_loss(val_batch[1], y_pred, val_batch[2])
                acc_value  = self.get_acc(val_batch[1], y_pred, val_batch[2])
                val_accura.append(acc_value)
                val_losses.append(loss_value)

            avg_epoch_loss_val = tf.reduce_mean(val_losses)
            avg_epoch_accu_val = tf.reduce_mean(val_accura)

            add_scalar_log(avg_epoch_loss_val, test_summary_writer, iter_count, 'loss')
            add_scalar_log(avg_epoch_accu_val, test_summary_writer, iter_count, 'accuracy')

            # =================================
            # ======== EARLY STOPPING =========
            # =================================
            if  avg_epoch_loss_val < best_loss:
                best_loss = avg_epoch_loss_val
                manager.save() # Saving weights
                early_stop_count = 0
            else:
                early_stop_count += 1

            if early_stop_count == patience:
                print('EARLY STOPPING ACTIVATED')
                break

    def predict_proba(self, test_batches, concat_batches=False):
        t0 = time.time()
        predictions = []
        true_labels = []

        for test_batch in test_batches:
            y_pred = self(test_batch[0], training=False)
            predictions.append(mask_pred(y_pred, test_batch[2]))
            true_labels.append(test_batch[1])

        t1 = time.time()
        print('runtime {:.2f}'.format((t1-t0)))
        if concat_batches:
            predictions = tf.concat(predictions, 0)
            true_labels = tf.concat(true_labels, 0)
        return predictions, true_labels


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
