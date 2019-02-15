from abc import ABC, abstractmethod
import tensorflow as tf

from darwin.operators.cx.basic import RandomCrossover
from darwin.objectives.basic import Scorer
from darwin.operators.mx.basic import RandomMutator
from structures.population import Population


class GeneticAlgorithm(ABC):

    @abstractmethod
    def optimize(self, population):
        pass


class BasicGeneticAlgorithm(GeneticAlgorithm):

    def __init__(self, num_iterations):
        self._num_iterations = num_iterations
        self._cx = RandomCrossover()
        self._mx = RandomMutator(0.1)
        self._scorer = Scorer(tf.reduce_max)
        self._num_individuals = None
        self._individual_length = None

    def stopping_criteria(self, iteration, *_):
        return tf.less(iteration, tf.constant(self._num_iterations, dtype=tf.int32))

    def loop_body(self, _, population, scores):
        children = self._mx.mutate(self._cx.crossover(
            population, self._num_individuals, self._individual_length),
            self._num_individuals,
            self._individual_length
        )
        children_scores = self._scorer.assign_scores(children)
        new_scores, top_idxs = tf.math.top_k(
            tf.concat(scores, children_scores), k=self._num_individuals
        )
        combined_pop = tf.concat(population, children)
        new_pop = tf.gather_nd(combined_pop, top_idxs)
        return new_pop, new_scores

    def optimize(self, population: Population):
        self._individual_length = population.individual_length
        self._num_individuals = population.num_individuals
        scores = self._scorer.assign_scores(population.data)
        return tf.while_loop(
            cond=self.stopping_criteria,
            body=self.loop_body,
            loop_vars=(0, population.data, scores)
        )
