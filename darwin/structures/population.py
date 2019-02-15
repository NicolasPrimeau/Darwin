
import tensorflow as tf


class Population:

    def __init__(self, num_inds, ind_length, ind_gen_fx=None):
        self._num_inds = num_inds
        self._ind_length = ind_length
        self._data = None
        if ind_gen_fx:
            self._data = tf.map_fn(
                fn=ind_gen_fx, elems=tf.range(num_inds, dtype=tf.int32)
            )

    def copy(self, data):
        pop = Population(self.num_individuals, self.individual_length)
        pop._data = data
        return pop

    def shuffle(self):
        return tf.random.shuffle(self._data)

    @property
    def num_individuals(self):
        return self._num_inds

    @property
    def individual_length(self):
        return self._ind_length

    @property
    def data(self):
        return self._data
