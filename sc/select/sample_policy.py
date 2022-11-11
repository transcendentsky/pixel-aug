"""
    A policy for choosing the best set of Max Similarity
        1. Genetic algorithm
        2. Random Traversing
        3. Traversing
"""
import numpy as np
import copy
import random
from tutils import tfilename, trans_args, trans_init
from tutils import CSVLogger
import argparse


# def sample_policy(points, matrix):
#     assert matrix.shape[0] == 150
#     mean_value = np.mean(matrix, axis=1)

def traversing(start=0, num=15, total=150):
    l = []
    for i in range(start, total):
        l += [i]
        l += traversing(start+1, num, total)


# class IdxIter:
#     def __init__(self, num=15, total=150):
#         self.num = num
#         self.total = total
#         self.indicater = np.array(np.arange(num))
#         self.start_state = np.array(np.arange(num))
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         is_add = [0] * self.num
#         is_forward = [0] * self.num # reset start state
#         # forward
#         for i in range(1, self.num+1):
#             if i == 1:
#                 self.indicater[-i] = self.indicater[-i] + 1
#             else:
#                 self.indicater[-i] = self.indicater[-i] + is_add[-i]
#
#             if self.indicater[-i] // (self.total - i + 1) == 1:
#                 self.start_state[-i] += 1
#                 if self.start_state[-i] // (self.total - i + 1) == 1:
#                     is_forward[-i] = 1
#                 is_add[-(i+1)] = 1
#                 # print("is forward")
#         if is_forward[0] == 1:
#             raise ValueError
#         # backward
#         for i in range(self.num):
#             if is_forward[i] == 1:
#                 self.start_state[i] = self.start_state[i-1] + 1
#                 self.indicater[i] = self.start_state[i]
#
#             # if self.indicater[i] // (self.total - i) == 1:
#             #     self.start_state[i] += 1
#             #     if self.start_state[i] // (self.total - i) == 1:
#             #         is_add[i-1] = 1
#             #
#             #     self.indicater[i] = self.start_state[i]
#             #
#             #     if is_add[i] > 0:
#             #         self.indicater[i] = self.indicater[i-1] + 1 - is_add[i]
#             #     # print("ss", i, self.indicater[i], self.indicater[i-1])
#             # # print("forward", is_add)
#             # assert self.indicater[i] < self.total, f"got {self.indicater}"
#         return self.indicater


def traver():
    """ traversing all choices,  """
    args = trans_args()
    logger, config = trans_init(args)

    def index_bin(num, total=150):
        s = str(bin(num))
        s_list = [int(si) for si in s[2:]]
        if len(s_list) < total:
            s_list = (total - len(s_list)) * [0] + s_list
        return s_list

    def decode(s):
        return np.where(np.array(s) == 1)[0]

    csvlogger = CSVLogger('traversing')

    matrix = np.load("data_max_list_all.npy")
    total = 150
    best_idx_15 = None
    best_fit_15 = 0
    best_idx_45 = None
    best_fit_45 = 0
    best_idx_75 = None
    best_fit_75 = 0

    for i in range(2**total):
        s = index_bin(i)
        assert len(s) == total
        if sum(s) == 15:
            gene = decode(s)
            assert len(gene) == 15
            fit = _fitness(gene, matrix=matrix)
            if fit > best_fit_15:
                best_fit_15 = fit
                gene.sort()
                best_idx_15 = gene
                best_idx_str = ''
                for idx in best_idx_15:
                    best_idx_str += f"{idx}, "
                logger.info(f"BEst gene 15: fit {best_fit_15}; idx {best_idx_str}")
                # csvlogger.record("15")
        elif sum(s) == 45:
            gene = decode(s)
            assert len(gene) == 45
            fit = _fitness(gene, matrix=matrix)
            if fit > best_fit_45:
                best_fit_45 = fit
                gene.sort()
                best_idx_45 = gene
                best_idx_str = ''
                for idx in best_idx_45:
                    best_idx_str += f"{idx}, "
                logger.info(f"BEst gene 45: fit {best_fit_45}; idx {best_idx_str}")
        elif sum(s) == 75:
            gene = decode(s)
            assert len(gene) == 75
            fit = _fitness(gene, matrix=matrix)
            if fit > best_fit_75:
                best_fit_75 = fit
                gene.sort()
                best_idx_75 = gene
                best_idx_str = ''
                for idx in best_idx_75:
                    best_idx_str += f"{idx}, "
                logger.info(f"BEst gene 75: fit {best_fit_75}; idx {best_idx_str}")


def test_one_fitness(gene=None):
    gene = [126, 128, 131, 132, 133, 134, 136, 139, 140, 141, 143, 144, 145, 146, 148]
    assert len(gene) == 15
    matrix = np.load("data_max_list_all.npy")
    fit = _fitness(gene, matrix=matrix)
    print(fit, gene)


def tmp():
    sift_tag = 'sift2'
    data_list = []
    for idx in range(150):
        # "/home1/quanquan/code/landmark/code/runs-ana/sift1/max_list/data_max_list_oneshot_0.npy"
        print(f"****  idx: {idx} ", end=" \r")
        # This is the
        data = np.load(tfilename(
            f'/home1/quanquan/code/landmark/code/runs-ana/' + sift_tag + f'/max_list/data_max_list_oneshot_{idx}.npy'))
        data = data[:, :, -1]
        data_list.append(data)
        # import ipdb; ipdb.set_trace()
    data_np = np.array(data_list)
    print("data np .shape", data_np.shape)
    np.save(tfilename("./", f"data_max_list_all.npy"), data_np)



def genetic_algo(matrix=None, num=5):
    # matrix = np.random.random((150, 10000))
    print(f"Search {num} samples")
    import ipdb; ipdb.set_trace()
    matrix = np.load("data_max_list_all.npy")
    num = num # 45 75
    geneticAlgo = GeneticAlgo(matrix, num)
    geneticAlgo.run()

####################################   Genetic Algorithm   ###################################################
def decode(x:list) -> list:
    def _decode(x:list):
        s = ''
        for xi in x:
            s += str(xi)
        i = int(s, 2)
        return i
    s = []
    for i in range(len(x)//8):
        xi = x[i*8:(i+1)*8]
        s.append(_decode(xi))
    return s

def encode(x:list) -> list:
    def _encode(x:int):
        s = str(bin(x))
        s_list = [int(si) for si in s[2:]]
        assert len(s_list) <= 8, f"Got {len(s_list)}"
        if len(s_list) < 8:
            s_list = (8-len(s_list)) * [0] + s_list
        return s_list
    s = []
    for xi in x:
        s += _encode(xi)
    return s


def fitness(gene, matrix):
    x = decode(gene)
    return _fitness(x, matrix)

class GeneticAlgo(object):
    def __init__(self, matrix, num):
        self.matrix = matrix
        self.num = num
        self.population = None

        self.pc = 0.8
        self.pm = 0.01
        self.max_gen = 10000
        self.pop_size = 1000
        self.code_length = num * 8
        # self.best_all = []
        self.best_in_iter = Chromosome([0,0,0,0,0,0,0,0], 0)

    def run(self):
        self.initialize_gene()
        best = self.find_best()
        # self.best_all.append(copy.deepcopy(best))
        for i in range(self.max_gen):
            self.cross()
            self.mutation()
            self.select()
            best = self.find_best()
            # self.best_all.append(copy.deepcopy(best))
            best_idx = decode(best.Genes)
            print(f"Iteration {i}:  Best fitness: {best.Fitness}, best idx {best_idx}")
            assert len(best_idx) == self.num

        # save
        # all_genes = [gene.Genes for gene in self.best_all]
        # all_fit = [gene.Fitness for gene in self.best_all]
        # np.save("all_genes.npy", all_genes)
        # np.save("all_fitness.npy", all_fit)

    def initialize_gene(self):
        """ initialize first genes """
        genes = []
        for i in range(self.pop_size):
            rand_idx = np.random.choice(np.arange(1, 150), self.num, replace=False).tolist()
            rand_idx.sort()
            # assert len(rand_idx) == self.num
            encoded_idx = encode(rand_idx)
            # assert len(encoded_idx) == self.num * 8
            genes.append(Chromosome(encoded_idx, fitness=fitness(encoded_idx, self.matrix)))
        self.population = genes
        # print("debug initalize gene ", self.population)
        # for g in self.population:
        #     print("genes ", g.Genes)
        assert len(self.population) == self.pop_size
        # print("len population", len(self.population))

    def find_best(self):
        """  Find the best in population  """
        all_fitness = np.array([gene.Fitness for gene in self.population])
        max_idx = all_fitness.argmax()
        best_gene = copy.deepcopy(self.population[max_idx])
        if best_gene.Fitness > self.best_in_iter.Fitness:
            self.best_in_iter = best_gene
        return best_gene

    def select(self):
        """  Roulette Selection  """
        # calculate fitness function
        all_fitness = np.array([gene.Fitness for gene in self.population])
        sum_fitness = all_fitness.sum()
        probs = all_fitness / sum_fitness
        select_score = np.array([p*np.random.random(self.pop_size) for p in probs])
        select_pop_idx = select_score.argmax(axis=0)
        self.population = [self.population[i] for i in select_pop_idx]

    def cross(self):
        """ Select 2 samples randomly, and cross-over at random location"""
        for k in range(len(self.population)//2):
            if self.pc > random.random():
                # Select 2 genes for crossing
                cross_idx = np.random.choice(np.arange(len(self.population)), 2, replace=False)
                cross_idx.sort()
                # Select 2 positions
                cross_loc = np.random.choice(np.arange(self.code_length), 2, replace=False)
                cross_loc.sort()
                # print("cross_loc", cross_loc)
                tmp_gene = copy.deepcopy(self.population[cross_idx[0]].Genes)
                gene_1 = self.population[cross_idx[0]].Genes
                gene_2 = self.population[cross_idx[1]].Genes
                gene_1 = gene_1[:cross_loc[0]] + gene_2[cross_loc[0]:cross_loc[1]] + gene_1[cross_loc[1]:]
                gene_2 = gene_2[:cross_loc[0]] + tmp_gene[cross_loc[0]:cross_loc[1]] + gene_2[cross_loc[1]:]
                # print(gene_1)
                assert len(gene_1) == self.code_length, f"Got {len(gene_1)}"
                assert len(gene_2) == self.code_length, f"Got {len(gene_2)}"
                self.population[cross_idx[0]].Genes = gene_1
                self.population[cross_idx[1]].Genes = gene_2

    def mutation(self):
        """  Mutation  """
        for i in range(len(self.population)):
            if self.pm > random.random():
                mute_loc = np.random.choice(np.arange(self.code_length), 1, replace=False)[0]
                self.population[i].Genes[mute_loc] = (self.population[i].Genes[mute_loc] + 1) % 2


class Chromosome:
    def __init__(self, genes, fitness):
        self.Genes = genes
        self.Fitness = fitness


def _fitness(x, matrix):
    v = matrix[x, :]
    vmax = v.max(axis=0)
    # assert len(vmax) == matrix.shape[1]
    return vmax.mean()


def debug_fitness(num=5):
    """  Random choose and get fitness  """
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ref", type=int, default=10)
    parser.add_argument("--num", type=int, default=5)
    args = trans_args(parser)
    logger, config = trans_init(args)

    matrix = np.load("data_max_list_all.npy")
    best_idx = None
    best_fit = 0
    num = num
    print("Note: args.ref: ", num)
    threshold = 0.6388
    save_idx_list = []
    save_fit_list = []

    threshold_iter = [100, 1000, 5000, 10000] # 100000

    logger.info(f"select num: {num}")
    for i in range(400000):
        if i in threshold_iter:
            best_idx_str = ''
            for idx in best_idx:
                best_idx_str += f"{idx},"
            logger.info(f"[***] Note: Iter {i}, best fit: {best_fit}, best idx: {best_idx_str}")
        gene = np.random.choice(np.arange(1, 150), num, replace=False)
        fit = _fitness(gene, matrix=matrix)
        if fit > threshold:
            # logger.info(f"[*] Greater than Threshold {threshold}, fit: {fit}, idx {gene}")
            save_fit_list.append(fit)
            gene.sort()
            save_idx_list.append(gene)
        if fit > best_fit:
            best_fit = fit
            gene.sort()
            best_idx = gene
            best_idx_str = ''
            for idx in best_idx:
                best_idx_str += f"{idx},"
            logger.info(f"[*] Iter {i}; Got new best! best fit:{best_fit}; idx {best_idx_str}")

    save_idx_list = np.array(save_idx_list)
    save_fit_list = np.array(save_fit_list)
    fit_sort = np.argsort(save_fit_list)[-10:]
    save_fit_list_10 = save_fit_list[fit_sort]
    save_idx_list_10 = save_idx_list[fit_sort]
    logger.info(save_fit_list_10)
    logger.info(save_idx_list_10)



if __name__ == '__main__':
    # tmp()
    # debug_fitness()
    import argparse
    from tutils import timer
    parser = argparse.ArgumentParser()
    # genetic_algo  / debug_fitness
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--func", default=None)
    args = parser.parse_args()
    funcname = args.func
    if funcname is not None:
        timer1 = timer()
        eval(funcname)(num=args.num)
        tt = timer1()
        print("Function: ", funcname)
        print("Time used: ", tt)
    # timer1 = timer()
    # timer1()
    # genetic_algo()
    # tt = timer1()
    # print("Time used ")