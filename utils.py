import os
import numpy as np
import math

OP_NAMES = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]


def get_num_classes(dataset):
    return 100 if dataset == 'cifar100' else 10 if dataset == 'cifar10' else 120


def to_string(ind):
    cell = ''
    node = 0
    for i in range(len(ind)):
        gene = ind[i]
        cell += '|' + OP_NAMES[gene] + '~' + str(node)
        node += 1
        if i == 0 or i == 2:
            node = 0
            cell += '|+'
    cell += '|'
    return cell


def better_fitness(fitness_1, fitness_2, maximization=True):
    if maximization:
        if fitness_1 > fitness_2:
            return True
    else:
        if fitness_1 < fitness_2:
            return True

    return False


# Returns 1 if x is equally preferable to y, 0 otherwise.
def equal_fitness(objective_value_x, objective_value_y):
    if objective_value_x == objective_value_y:
        return True
    return False


def create_exp_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    print('Experiment dir : {}'.format(path))
