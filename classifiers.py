import tensorflow as tf
from phased2 import PhasedLSTM 
from phased import PhasedLSTM as OLDPhasedLSTM
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import LayerNormalization, LSTMCell, RNN
import numpy as np
import time


def fix_tensor(x, mask):
    valid_mask = tf.cast(tf.reduce_sum(mask, 1), dtype=tf.bool)
    x = tf.boolean_mask(x, valid_mask)
    return x

def mask_pred(y_pred, mask):
    valid_mask = tf.cast(tf.reduce_sum(mask, 1), dtype=tf.bool)
    
    mask = tf.boolean_mask(mask, valid_mask)
    y_pred = tf.boolean_mask(y_pred, valid_mask)

    last = tf.cast(tf.reduce_sum(mask, 1), tf.int32)-1
    last = tf.stack([tf.range(last.shape[0]), last], 1)

    last_probas  = tf.gather_nd(y_pred, last)
    last_probas  = tf.expand_dims(last_probas, 2)
    tensor_last  = tf.tile(last_probas, [1,1,y_pred.shape[1]])
    inverse_mask = tf.cast(tf.where(mask == 0, 1, 0), dtype=tf.float32)
    tensor_last  = tf.transpose(tensor_last, [0,2,1]) * tf.expand_dims(inverse_mask, 2)
    tensor_first = y_pred * tf.expand_dims(mask, 2)
    return tensor_first + tensor_last

def add_scalar_log(value, writer, step, name):
    with writer.as_default():
        tf.summary.scalar(name, value, step=step)

# =====================================================
# ================= PHASED CLASSIFIER =================
# =====================================================
class PhasedClassifier(tf.keras.Model):

    def __init__(self, units, n_classes, dropout=0.5, name='phased', use_old=False):
        super(PhasedClassifier, self).__init__()
        self._cells = []
        self._units = units
        self._name  = name

        if use_old:
            self.plstm_0 = OLDPhasedLSTM(self._units, name='rnn_0')
            self.plstm_1 = OLDPhasedLSTM(self._units, name='rnn_1')
        else:    
            self.plstm_0 = PhasedLSTM(self._units, name='rnn_0')
            self.plstm_1 = PhasedLSTM(self._units, name='rnn_1')

        self.fc = tf.keras.layers.Dense(n_classes,
                                        activation='softmax',
                                        dtype='float32')
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.norm_layer = LayerNormalization()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    @tf.function
    def call(self, inputs, times, training=False, normalize=False):
        states_0 = (tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units]))

        states_1 = (tf.zeros([inputs.shape[0], self._units]),
                    tf.zeros([inputs.shape[0], self._units]))

        initial_state = (states_0, states_1)

        if normalize:
            min_values = tf.expand_dims(tf.reduce_min(inputs, axis=1), 1)
            max_values = tf.expand_dims(tf.reduce_max(inputs, axis=1), 1)
            inputs = (inputs - min_values) / (max_values - min_values)
            
        x_t = tf.transpose(inputs, [1, 0, 2])
        t_t = tf.transpose(times, [1, 0])
        
        time_steps = tf.shape(x_t)[0]

        def compute(i, cur_state, out):
            output_0, cur_state0 = self.plstm_0((x_t[i], t_t[i]), states=cur_state[0])
            output_0 = self.norm_layer(output_0)
            output_1, cur_state1 = self.plstm_1((output_0, t_t[i]), states=cur_state[1])
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


        # Define checkpoints
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                   model=self,
                                   optimizer=self.optimizer)

        manager = tf.train.CheckpointManager(ckpt,
                                             save_path+'/{}'.format(self._name)+'/ckpts',
                                             max_to_keep=1)
        # Training variables
        best_loss = 9999999
        early_stop_count = 0
        iter_count = 0 # each forward pass is an iteration
        curr_eta = 0.
        for epoch in range(epochs):
            print('[INFO] RUNNING EPOCH {}/{} - EARLY STOP COUNTDOWN: {} - ETA:{:.1f} sec'.format(epoch, epochs, patience-early_stop_count, curr_eta), end='\r')
            t0 = time.time()
            for train_batch in train:
                with tf.GradientTape() as tape:
                    y_pred = self(train_batch[0], train_batch[0][...,0], training=True, normalize=True)
                    loss_value = self.get_loss(train_batch[1], y_pred, train_batch[2])
                    acc_value = self.get_acc(train_batch[1], y_pred, train_batch[2])

                add_scalar_log(acc_value, train_summary_writer, iter_count, 'accuracy')
                add_scalar_log(loss_value, train_summary_writer, iter_count, 'loss')

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                iter_count+=1

            
            val_losses = []
            val_accura = []
            for val_batch in val:
                y_pred = self(val_batch[0], val_batch[0][...,0], normalize=True)
                loss_value = self.get_loss(val_batch[1], y_pred, val_batch[2])
                acc_value  = self.get_acc(val_batch[1], y_pred, val_batch[2])

                val_losses.append(loss_value)
                val_accura.append(acc_value)

            t1 = time.time()
            curr_eta = (t1 - t0)*(epochs-epoch+1)

            avg_epoch_loss_val = tf.reduce_mean(val_losses)
            avg_epoch_accu_val = tf.reduce_mean(val_accura)

            add_scalar_log(avg_epoch_loss_val, test_summary_writer, iter_count, 'loss')
            add_scalar_log(avg_epoch_accu_val, test_summary_writer, iter_count, 'accuracy')

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
            y_pred = self(test_batch[0], test_batch[0][...,0], normalize=True)
            loss_value = self.get_loss(test_batch[1], y_pred, test_batch[2])
            test_losses.append(loss_value)
            predictions.append(mask_pred(y_pred, test_batch[2]))
            true_labels.append(fix_tensor(test_batch[1], test_batch[2]))

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

    def __init__(self, units, n_classes, dropout=0.5, name='phased'):
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
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        
    @tf.function
    def call(self, inputs, times, training=False):
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

        return tf.transpose(out.stack(), [1,0,2])


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

        # Define checkpoints
        ckpt = tf.train.Checkpoint(step=tf.Variable(1),
                                   model=self,
                                   optimizer=self.optimizer)

        manager = tf.train.CheckpointManager(ckpt,
                                             save_path+'/{}'.format(self._name)+'/ckpts',
                                             max_to_keep=1)
        # Training variables
        best_loss = 9999999
        early_stop_count = 0
        iter_count = 0 # each forward pass is an iteration
        curr_eta = 0.
        for epoch in range(epochs):
            print('[INFO] RUNNING EPOCH {}/{} - EARLY STOP COUNTDOWN: {} - ETA:{:.1f} sec'.format(epoch, epochs, patience-early_stop_count, curr_eta), end='\r')
            t0 = time.time()
            for train_batch in train:
                with tf.GradientTape() as tape:
                    y_pred = self(train_batch[0], train_batch[2], training=True)
                    loss_value = self.get_loss(train_batch[1], y_pred, train_batch[2])
                    acc_value = self.get_acc(train_batch[1], y_pred, train_batch[2])
                
                add_scalar_log(acc_value, train_summary_writer, iter_count, 'accuracy')
                add_scalar_log(loss_value, train_summary_writer, iter_count, 'loss')

                grads = tape.gradient(loss_value, self.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
                iter_count+=1 

            val_losses = []
            val_accura = []
            for val_batch in val:
                y_pred = self(val_batch[0], val_batch[2])
                loss_value = self.get_loss(val_batch[1], y_pred, val_batch[2])
                acc_value  = self.get_acc(val_batch[1], y_pred, val_batch[2])
                val_accura.append(acc_value)
                val_losses.append(loss_value)

            t1 = time.time()
            
            curr_eta = (t1 - t0)*(epochs-epoch+1)

            avg_epoch_loss_val = tf.reduce_mean(val_losses)
            avg_epoch_accu_val = tf.reduce_mean(val_accura)

            add_scalar_log(avg_epoch_loss_val, test_summary_writer, iter_count, 'loss')
            add_scalar_log(avg_epoch_accu_val, test_summary_writer, iter_count, 'accuracy')

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
            predictions.append(mask_pred(y_pred, test_batch[2]))
            true_labels.append(fix_tensor(test_batch[1], test_batch[2]))
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