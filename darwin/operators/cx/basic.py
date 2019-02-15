from abc import ABC, abstractmethod

import tensorflow as tf


class CrossoverOperator(ABC):

    @abstractmethod
    def crossover(self, population, num_individuals, individual_length):
        pass


class RandomSelector:

    def get_parents(self, population, num_individuals, individual_length):

        def pair_up_parents(idx):
            return tf.stack((population[idx], population[idx + 1]))

        return tf.map_fn(
            fn=pair_up_parents,
            elems=tf.range(0, num_individuals, 2)
        )


class RandomBinarySplicer:

    def cross_parents(self, parents, parent_length):
        options = tf.reshape(parents, (None, 2))

        def choose_func(option):
            return tf.random.shuffle(option)[0]

        return tf.map_fn(fn=choose_func, elems=options)


class RandomCrossover(CrossoverOperator):

    def __init__(self):
        self._selector = RandomSelector()
        self._splicer = RandomBinarySplicer()

    def crossover(self, population, num_individuals, individual_length):
        return tf.map_fn(
            fn=self._splicer.cross_parents,
            elems=(
                self._selector.get_parents(population, num_individuals, individual_length),
                individual_length
            )
        )
