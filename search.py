import random
import numpy as np
from nats_bench import create
from random import choice
import argparse
import torch
import logging
import os
import sys
import pickle
import time

from foresight.dataset import get_cifar_dataloaders
from foresight.pruners import predictive
from foresight.weight_initializers import init_net
from foresight.models.nasbench2 import get_model_from_arch_str

from utils import *


class NAS():
    def __init__(self, method, population_size, number_of_generations, dataset, objective, variation_type, gom_metric,
                 save_dir, device, train_loader):
        self.method = method
        self.number_of_generations = number_of_generations
        self.number_of_evaluations = 0
        self.population_size = population_size
        self.number_of_variables = 6  
        self.variable_range = [5] * 6
        self.objective = objective
        self.variation_type = variation_type 
        self.gom_metric = gom_metric
        self.save_dir = save_dir
        self.device = device
        self.dataset = dataset
        self.datasets = ['cifar10', 'cifar100', 'ImageNet16-120'] if dataset == 'cifar10' else [dataset]
        self.train_loader = train_loader
        self.total_train_and_valid_time = 0

        self.hp = 12
        self.api = create("benchmark/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)
        self.op_names = ["none", "skip_connect", "nor_conv_1x1", "nor_conv_3x3", "avg_pool_3x3"]
        self.selection_size = population_size
        self.tournament_size = 4 if method in ['SF-GOMENAS', 'ENAS', 'TF-ENAS'] else 2
        self.links = []
        self.archive_objective = []
        self.archive_var = []
        self.archive_testacc = {
            'cifar10': [],
            'cifar100': [],
            'ImageNet16-120': []
        }
        self.log_run = {
            'pop-fitness': [],
            'pop': [],
            'time': [],
        }

    def initialize_population(self):
        pop = np.random.randint(5, size=(self.population_size, self.number_of_variables))
        return pop

    def evaluate_arch(self, ind, metric, dataset):
        if metric in ['synflow']:
            arch_str = to_string(ind)
            net = get_model_from_arch_str(arch_str, get_num_classes(dataset))
            init_net(net, 'kaiming', 'zero')

            measure = predictive.find_measures(net,
                                               self.train_loader,
                                               ('random', 1, get_num_classes(dataset)),
                                               self.device,
                                               measure_names=[metric])

            res = measure[metric]

            if self.method in ['TF-ENAS']:
                time_ = time.time()
                self.log_run['time'].append(time_)
                self.number_of_evaluations += 1
                self.archive_check(ind, res)

        else:
            if dataset == 'cifar10' and metric.startswith(("train", "val")):
                dataset = 'cifar10-valid'

            ind_str = to_string(ind)
            index = self.api.query_index_by_arch(ind_str)

            if metric == 'valid-accuracy':
                info = self.api.get_more_info(index=index, dataset=dataset, hp=self.hp, is_random=False)
                res = info[metric]
                time_ = time.time()
                self.total_train_and_valid_time += info['train-all-time'] + info['valid-per-time']
                self.log_run['time'].append(time_ + self.total_train_and_valid_time)
                self.number_of_evaluations += 1
                self.archive_check(ind, res)
            else: 
                info = self.api.get_more_info(index=index, dataset=dataset, hp=200, is_random=False)
                res = info[metric]
        return res

    def archive_check(self, ind, fitness):
        if len(self.archive_objective) == 0 or better_fitness(fitness, self.archive_objective[-1]):
            self.archive_objective.append(fitness)
            self.archive_var.append(ind)

            for dat in self.datasets:
                testacc = self.evaluate_arch(ind, 'test-accuracy', dat)
                self.archive_testacc[dat].append(testacc)

        else:
            self.archive_objective.append(self.archive_objective[-1])
            for dat in self.datasets:
                self.archive_testacc[dat].append(self.archive_testacc[dat][-1])

    def evaluate_population(self, pop):
        test_accuracy, max_testacc = {}, {}
        values = np.array([self.evaluate_arch(ind, self.objective, self.dataset) for ind in pop])
        return values

    def tournament_selection(self, pop, pop_fitness):
        num_individuals = len(pop)
        indices = np.arange(num_individuals)
        selected_indices = []

        while len(selected_indices) < self.selection_size:
            np.random.shuffle(indices)

            for i in range(0, num_individuals, self.tournament_size):
                best_idx = i
                for idx in range(1, self.tournament_size):
                    if better_fitness(pop_fitness[indices[i + idx]], pop_fitness[indices[best_idx]]):
                        best_idx = i + idx
                selected_indices.append(indices[best_idx])

        selected_indices = np.array(selected_indices)

        return selected_indices

    def UX(self, pop):
        num_individuals = len(pop)
        num_parameters = len(pop[0])
        indices = np.arange(num_individuals)
        np.random.shuffle(indices)
        offspring = []

        for i in range(0, num_individuals, 2):
            idx1 = indices[i]
            idx2 = indices[i + 1]
            offspring1 = list(pop[idx1])
            offspring2 = list(pop[idx2])

            for idx in range(0, num_parameters):
                r = np.random.rand()
                if r < 0.5:
                    temp = offspring2[idx]
                    offspring2[idx] = offspring1[idx]
                    offspring1[idx] = temp

            offspring.append(offspring1)
            offspring.append(offspring2)

        offspring = np.array(offspring)
        return offspring

    def gom(self, pop, pop_fitness=None):
        offspring = pop.copy()
        if self.method in ['GOMENAS']:
            offspring_fitness = pop_fitness.copy()

        for i in range(len(pop)):
            random.shuffle(self.links)
            if self.method in ['GOMENAS']:
                fitness_exist = pop_fitness[i]
            else:
                fitness_exist = self.evaluate_arch(pop[i], self.gom_metric, self.dataset)
            for link in self.links:
                if len(link) == self.number_of_variables:
                    continue
                ind_rand_index = choice([j for j in range(len(pop)) if j not in [i]])
                ind_rand = pop[ind_rand_index].copy()
                ind_new = offspring[i].copy()
                ind_new[link] = ind_rand[link]

                if np.array_equal(ind_new, pop[i]) == False:
                    fitness_new = self.evaluate_arch(ind_new, self.gom_metric, self.dataset)

                    if better_fitness(fitness_new, fitness_exist):
                        offspring[i] = ind_new.copy()
                        fitness_exist = fitness_new
                        if self.method in ['GOMENAS']:
                            offspring_fitness[i] = fitness_new

        if self.method in ['GOMENAS']:
            return offspring, offspring_fitness
        return offspring

    def converge_pop(self, pop):
        if np.all(pop == pop[0]):
            return True
        return False

    def univariate_learn_linkage(self, pop):
        self.FOSs_length = self.number_of_variables
        self.FOSs = np.zeros((self.FOSs_length, 1), dtype=np.int32)
        self.FOSs_number_of_indices = np.zeros(self.FOSs_length, dtype=np.int32)

        order = np.random.permutation(self.number_of_variables)
        FOSs_index = 0
        for i in range(self.number_of_variables):
            self.FOSs[FOSs_index][0] = order[FOSs_index]
            self.FOSs_number_of_indices[FOSs_index] = 1
            FOSs_index += 1

    def solve(self):
        start = time.time()
        self.log_run['time'].append(start)
        pop = self.initialize_population()
        pop_fitness = self.evaluate_population(pop)
        
        self.univariate_learn_linkage(pop)
        lks = []
        for i in range(self.FOSs_length):
            l = []
            for j in range(self.FOSs_number_of_indices[i]):
                l.append(self.FOSs[i][j])
            lks.append(l)
        self.links = lks
     
        for i in range(self.number_of_generations):
            if self.method in ['SF-GOMENAS']:
                offspring = self.gom(pop)
            elif self.method in ['GOMENAS']:
                offspring, offspring_fitness = self.gom(pop, pop_fitness)
            else:
                offspring = self.UX(pop)

            if self.method in ['GOMENAS']:
                offspring_indices = self.tournament_selection(offspring, offspring_fitness)
                pop = offspring[offspring_indices, :]
                pop_fitness = offspring_fitness[offspring_indices]

            else:
                offspring_fitness = self.evaluate_population(offspring)
                pool = np.vstack((pop, offspring))
                pool_fitness = np.hstack((pop_fitness, offspring_fitness))
                pool_indices = self.tournament_selection(pool, pool_fitness)
                pop = pool[pool_indices, :]
                pop_fitness = pool_fitness[pool_indices]

            print("#Gen {}:".format(i + 1))
            self.log_run['pop-fitness'].append(pop_fitness)
            self.log_run['pop'].append(pop)

            if (self.converge_pop(pop)):
                break

        end = time.time()
        print("#Result:")
        print(pop)
        print(pop_fitness)

        self.log_run['archive-objective'] = self.archive_objective
        self.log_run['archive-testacc'] = self.archive_testacc
        self.log_run['time'].append(end + self.total_train_and_valid_time)

        pickle_out = open(os.path.join(self.save_dir, f'seed_{seed}.pickle'), "wb")
        pickle.dump(self.log_run, pickle_out)
        pickle_out.close()

def parse_arguments():
    parser = argparse.ArgumentParser("Gene-pool Optimal Mixing Evolutionary Neural Architecture Search")
    parser.add_argument('--method', type=str, default='SF-GOMENAS',
                        help='method [GOMENAS, SF-GOMENAS, ENAS, TF-ENAS')
    parser.add_argument('--n_runs', type=int, default=30, help='number of runs')
    parser.add_argument('--pop_size', type=int, default=20, help='population size of networks')
    parser.add_argument('--n_gens', type=int, default=50, help='number of generations')
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset [cifar10, cifar100, ImageNet16-120]')
    parser.add_argument('--continue_run', type=int, default=0, help='continue from the last run')

    parser.add_argument('--init_w_type', type=str, default='kaiming',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='zero',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to work on')
    parser.add_argument('--num_data_workers', type=int, default=2, help='number of workers for dataloaders')
    parser.add_argument('--dataload', type=str, default='random', help='random or grasp supported')
    parser.add_argument('--dataload_info', type=int, default=1,
                        help='number of batches to use for random dataload or number of samples per class for grasp dataload')


    args = parser.parse_args()
    args.objective = 'valid-accuracy' if args.method in ['ENAS', 'GOMENAS', 'SF-GOMENAS'] else 'synflow'
    args.variation_type = 'UX' if args.method in ['ENAS', 'TF-ENAS'] else 'gom'

    if args.method in ['SF-GOMENAS']:
        args.gom_metric = 'synflow'
    elif args.method in ['GOMENAS']:
        args.gom_metric = 'valid-accuracy'
    else:
        args.gom_metric = None

    args.save = 'experiment/{}/{}'.format(args.method, args.dataset)
    create_exp_dir(args.save)
    args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

    return args


if __name__ == '__main__':
    args = parse_arguments()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.dataset == "ImageNet16-120":
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                          args.num_data_workers, resize=None, datadir='benchmark/')
    else:
        train_loader, test_loader = get_cifar_dataloaders(args.batch_size, args.batch_size, args.dataset,
                                                          args.num_data_workers)

    for seed in range(args.continue_run, args.n_runs):
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logging.info('seed: {}'.format(seed))
        problem = NAS(args.method, args.pop_size, args.n_gens, args.dataset,
                      args.objective, args.variation_type, args.gom_metric,
                      args.save, args.device, train_loader)

        problem.solve()



