import tensorflow as tf
from tensorflow.compat.v1 import constant_initializer

def _exponential_initializer(min, max, dtype=None):
    def in_func(shape, dtype=dtype):
        initializer = tf.random_uniform_initializer(
                        tf.math.log(1.0),
                        tf.math.log(100.0)
                        )
        return tf.math.exp(initializer(shape))
    return in_func

class PhasedLSTM(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 leak_rate=0.001,
                 ratio_on=0.1,
                 period_init_min=1.0,
                 period_init_max=1000.0,
                 rec_activation = tf.math.sigmoid,
                 out_activation = tf.math.tanh,
                 name='plstm'):
        super(PhasedLSTM, self).__init__(name=name)
        self.units = units
        self._leak = leak_rate
        self._ratio_on = ratio_on
        self._rec_activation = rec_activation
        self._out_activation = out_activation
        self.period_init_min = period_init_min
        self.period_init_max = period_init_max

    def _mod(self, x, y):
        """Modulo function that propagates x gradients."""
        return tf.stop_gradient((x%y) - x) + x

    def _get_cycle_ratio(self, time, phase, period):
        """Compute the cycle ratio in the dtype of the time."""
        phase_casted = tf.cast(phase, dtype=time.dtype)
        period_casted = tf.cast(period, dtype=time.dtype)
        time = tf.reshape(time, [time.shape[0],1])
        shifted_time = time - phase_casted
        cycle_ratio = self._mod(shifted_time, period_casted)
        cycle_ratio = self._mod(shifted_time, period_casted) / period_casted
        return tf.cast(cycle_ratio, dtype=tf.float32)

    def build(self, input_shape):
        self.kernel = self.add_weight(
                        name="kernel",
                        shape=[4, input_shape[-1]+self.units, self.units],
                        dtype=tf.float32,
                        initializer=tf.keras.initializers.GlorotNormal(),
                        trainable=True)

        self.period = self.add_weight(
                        name="period",
                        shape=[self.units],
                        dtype=tf.float32,
                        initializer=_exponential_initializer(
                                            self.period_init_min,
                                            self.period_init_max),
                        trainable=True)
        self.phase = self.add_weight(name="phase",
                                     shape=[self.units],
                                     initializer=tf.random_uniform_initializer(
                                                         0.0,
                                                         self.period),
                                     trainable=True)
        self.ratio_on = self.add_weight(name="ratio_on",
                                        shape=[self.units],
                                        initializer=constant_initializer(self._ratio_on),
                                        trainable=True)
    @tf.function
    def call(self, input, times, states):
        cg, fg, sg, og = tf.unstack(self.kernel, 4, axis=0)
        c_prev, h_prev = states
        concat_input = tf.concat([input, h_prev], axis=1)
        # =================================
        # RECURRENT GATES
        # =================================
        ca = tf.linalg.matmul(concat_input, cg)
        ca = self._out_activation(ca)

        fa = tf.linalg.matmul(concat_input, fg)
        fa = self._rec_activation(fa)

        sa = tf.linalg.matmul(concat_input, sg)
        sa = self._rec_activation(sa)

        oa = tf.linalg.matmul(concat_input, og)
        oa = self._rec_activation(oa)

        # =================================
        # CANDIDATE CELL AND HIDDEN STATE
        # =================================
        cell_state = fa*c_prev + sa*ca
        hidden_state = oa*self._rec_activation(cell_state)
        # =================================
        # TIME GATE
        # =================================
        cycle_ratio = self._get_cycle_ratio(times, self.phase, self.period)

        k_up = 2 * cycle_ratio / self.ratio_on
        k_down = 2 - k_up
        k_closed = self._leak * cycle_ratio

        k = tf.where(cycle_ratio < self.ratio_on, k_down, k_closed)
        k = tf.where(cycle_ratio < 0.5 * self.ratio_on, k_up, k)

        # =================================
        # UPDATE STATE USING TIME GATE VALUES
        # =================================
        new_c = k * cell_state + (1 - k) * c_prev
        new_h = k * hidden_state + (1 - k) * h_prev

        return new_h, (new_c, new_h)
