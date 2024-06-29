import copy
import os
import random
import numpy as np

from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Data.Benchmark import Benchmark
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans, EvalCoreset
from Source.Util.parallel import start_tasks
from Source.Util.util import unique, str_to_int_list
from Source.Util.util import get_root_path
import argparse


class SequenceGen:
    def __init__(self, seed):
        self.rnd = random.Random()
        self.rnd.seed(seed)

    def next_seq(self, step=45):
        rand_seq = [self.rnd.randint(0, 123) for _ in range(step)]
        return rand_seq


class RandomSearch:
    cpu_count = os.cpu_count()

    @staticmethod
    def trans(temp: ProSeqSiz, seq: list[int]) -> tuple:
        pro_seq2 = copy.deepcopy(temp)
        pro_seq2.sequence = seq
        return pro_seq2.export()

    def __init__(self, benchmark: Benchmark, num_of_program: int, episode: int):
        self.benchmark = benchmark
        self.benchmark.cut(num_of_program)
        self.seqs = []
        self.coreset = []
        self.matrix = []
        self.coreset_index = []
        self.episode = episode
        self.identifier = f'{benchmark.name}_{num_of_program}'
        prefix = f'{get_root_path()}/OtherWork/SearchBasedCoreset'
        self.candidate_path = f'{prefix}/candidate_{self.identifier}'
        self.matrix_path = f'{prefix}/matrix_{self.identifier}'
        self.coreset_path = f'{prefix}/coreset_{self.identifier}'

    @staticmethod
    def search_one(pss, seed, episode, sql_list):
        env = EnvManager()
        seq_gen = SequenceGen(seed)
        res = []
        for _ in range(episode):
            seq = seq_gen.next_seq()
            new_pss = Trans.copy_and_reset(pss, seq)
            cut_pss, size = env.evaluate_for_mp_with_cut(new_pss, sql_list)
            res.append(ProSeqSiz.import_pss(cut_pss, size))
        min_pss = min(res)
        return min_pss

    def random_search(self):
        print('Now, run random search on programs.')
        if os.path.exists(self.candidate_path):
            return

        args = [(pss, idx, self.episode) for idx, pss in enumerate(self.benchmark.pss_list)]
        res = start_tasks(RandomSearch.search_one, args, desc='Random Search')
        for pss in res:
            self.seqs.append(pss.sequence)
        self.seqs = unique(self.seqs)

        with open(self.candidate_path, 'w') as f:
            f.write(f'')
            for seq in self.seqs:
                f.write(f'{seq}\n')

    @staticmethod
    def get_matrix_row(pss, seqs, sql_list):
        env = EnvManager()
        res1 = [env.evaluate_for_mp(Trans.copy_and_reset(pss, seq), sql_list) for seq in seqs]
        res2 = env.evaluate_for_mp(Trans.copy_and_reset(pss, []), sql_list)
        res = [e / res2 if res2 != 0 else 1.0 for e in res1]
        return res

    def get_matrix(self):
        if os.path.exists(self.matrix_path):
            self.matrix = np.loadtxt(self.matrix_path, delimiter=',', dtype=float)
        else:
            args = [(pss, self.seqs) for pss in self.benchmark.pss_list]
            res = start_tasks(RandomSearch.get_matrix_row, args, desc='Matrix')
            m = []
            for row in res:
                m.append(row)

            self.matrix = np.array(m)
            self.matrix = np.array(self.matrix).T
            np.savetxt(self.matrix_path, self.matrix, fmt='%f', delimiter=',')

    def gen_coreset(self, k=50):
        print('Now, generate coreset.')
        with open(self.candidate_path, 'r') as f:
            for line in f:
                self.seqs.append(str_to_int_list(line))
        self.get_matrix()
        for it in range(k):
            mx, val = 0, -float("inf")
            for idx, seq in enumerate(self.seqs):
                if idx in self.coreset_index:
                    continue
                cp_coreset_index = copy.deepcopy(self.coreset_index)
                cp_coreset_index.append(idx)
                size_sum = np.sum(np.max(self.matrix[cp_coreset_index], axis=0))
                if size_sum > val:
                    mx, val = idx, size_sum
            self.coreset_index.append(mx)

        with open(self.coreset_path, 'w') as f:
            for seq_index in self.coreset_index:
                f.write(f'{self.seqs[seq_index]}\n')
                self.coreset.append(self.seqs[seq_index])


def parse():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument('-e', '--episode', required=True, type=int, help='Number of pass sequences')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    search_benchmark = Benchmark('__it_train', '__it_train')
    RS = RandomSearch(
        search_benchmark,
        500,
        args.episode,
    )
    RS.random_search()
    RS.gen_coreset()
