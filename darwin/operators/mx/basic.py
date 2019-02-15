from abc import abstractmethod, ABC

import tensorflow as tf


class Mutator(ABC):

    @abstractmethod
    def mutate(self, population, num_individuals, individual_length):
        pass


class IndividualRandomMutator:

    def mutate(self, individual, ind_size):
        index = tf.random.shuffle(tf.range(ind_size)[0])
        return tf.concat((
            individual[:index],
            tf.random.uniform(1)[0],
            individual[index + 1:]
        ))


class RandomMutator(Mutator):

    def __init__(self, mx_prob):
        self._mx_prob = mx_prob
        self._mutator = IndividualRandomMutator()

    def mutate(self, population, num_individuals, individual_length):

        def random_mutation(individual):
            return tf.cond(
                pred=tf.less(tf.random.uniform(1)[0], self._mx_prob),
                true_fn=lambda: self._mutator.mutate(individual, individual_length),
                false_fn=individual
            )

        return tf.map_fn(
            fn=random_mutation,
            elems=population
        )
