import copy
import os
import random
import numpy as np

from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Data.Benchmark import Benchmark
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans, EvalCoreset
from Source.OtherWork.CoverSet.algorithms.metaheuristics import SGA
from Source.Solver.CoresetGen.CoresetGen import CoresetGen
from Source.Util.parallel import start_tasks
from Source.Util.util import unique, str_to_int_list, read_coreset_to_file, get_root_path, save_coreset_to_file
import argparse


class GenCoverset:

    def __init__(self, benchmark: Benchmark):
        self.benchmark = benchmark
        self.seqs = []
        self.identifier = f'{benchmark.name}'
        self.prefix = f'{get_root_path()}/OtherWork/CoverSet'
        self.candidate_path = f'{self.prefix}/candidate_{self.identifier}'

    @staticmethod
    def search_one(pss, seed, sql_list):
        file_path = f'{get_root_path()}/OtherWork/CoverSet/seq_{seed}.txt'
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                sequence = str_to_int_list(f.read())
            return ProSeqSiz.import_pss(Trans.copy_and_reset(pss, sequence))

        env = EnvManager()

        sga = SGA(
            generations=35,  # 35
            population=35,  # 35
            cr=0.9,
            m=0.1,
            param_m=1.0,
            param_s=1,
            crossover='exponential',
            mutation='polynomial',
            selection='tournament',
            seed=seed,
            dimension=45,  # ?
            env=env,
            pss=pss,
        )

        pss_list, temp_sql_list = sga.run()
        min_pss = min(pss_list)

        with open(file_path, 'w') as f:
            f.write(f'{min_pss.sequence}\n')
        return min_pss

    def genetic(self):
        args = [(pss, idx, ) for idx, pss in enumerate(self.benchmark.pss_list)]
        res = start_tasks(GenCoverset.search_one, args, desc='Genetic')
        self.benchmark.pss_list = res

        for pss in res:
            self.seqs.append(pss.sequence)
        self.seqs = unique(self.seqs)

        with open(self.candidate_path, 'w') as f:
            f.write(f'')
            for seq in self.seqs:
                f.write(f'{seq}\n')

    def sequence_reduction(self, pro_seq_siz: ProSeqSiz, sql_list):
        change = True
        sequence = pro_seq_siz.sequence
        env = EnvManager()
        cur_size = env.evaluate_for_mp(pro_seq_siz.export(), sql_list)
        while change is True:
            change = False
            for i in range(len(sequence)):
                cur_seq = copy.deepcopy(sequence)
                cur_seq.pop(i)

                tmp = Trans.copy_and_reset(pro_seq_siz, cur_seq)
                size = env.evaluate_for_mp(tmp, sql_list)
                if size <= cur_size:
                    size, sequence = cur_size, cur_seq
                    change = True
                    break
        return ProSeqSiz.import_pss(Trans.copy_and_reset(pro_seq_siz, sequence), cur_size)

    def gen_coreset(self):
        args = [(pss, ) for pss in self.benchmark.pss_list]
        self.benchmark.pss_list = start_tasks(self.sequence_reduction, args, desc='Seq Reduction')
        self.seqs = [pss.sequence for pss in self.benchmark.pss_list]
        save_coreset_to_file(self.seqs, f'{self.prefix}/reduced_seqs.txt')
        coreset_gen = CoresetGen(self.benchmark)
        coreset_gen.gen_coreset(f'{self.prefix}/gens_coreset.txt')


if __name__ == '__main__':
    genetic_search = GenCoverset(
        Benchmark('__it_train', '__it_train'),
    )
    genetic_search.genetic()
    genetic_search.gen_coreset()
