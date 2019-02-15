import tensorflow as tf


class Scorer:

    def __init__(self, ind_scorer_fn):
        self._ind_scorer_fn = ind_scorer_fn

    def assign_scores(self, population):
        return tf.map_fn(
            fn=self._ind_scorer_fn, elems=population
        )
