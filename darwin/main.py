from darwin.algorithms.basic import BasicGeneticAlgorithm
from darwin.structures.individual import RandomIndividualFactory
from darwin.structures.population import Population
import tensorflow as tf


def main():
    algo = BasicGeneticAlgorithm(1000)
    factory = RandomIndividualFactory(10)
    population = Population(100, 10, factory.build)
    optimized, scores = algo.optimize(population)

    with tf.Session() as sess:
        sess.run()


if __name__ == '__main__':
    main()


