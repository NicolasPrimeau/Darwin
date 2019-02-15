
import tensorflow as tf


class Individual:

    def __init__(self, length, init=0.0):
        self._data = tf.fill([length], init)

    def apply_op(self, fx):
        return fx(self._data)


class RandomIndividualFactory:

    def __init__(self, length):
        self._length = length

    def build(self, _):
        return tf.random.uniform((self._length, ))
