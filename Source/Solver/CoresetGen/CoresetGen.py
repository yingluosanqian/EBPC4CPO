import sys
sys.path.append('./../../../')
import copy
import random

import math
import numpy as np

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import save_coreset_to_file, get_arith_mean


class CoresetGen:
    def __init__(self, benchmark: Benchmark, seqs: list[list[int]] = None):
        self.benchmark = benchmark
        self.seqs = [pss.sequence for pss in self.benchmark.pss_list]
        if seqs is not None:
            self.seqs = seqs
            assert len(self.benchmark.pss_list) == len(self.seqs)
            for i in range(len(self.seqs)):
                self.benchmark.pss_list[i].sequence = self.seqs[i]
        self.labels = []

        def get_benchmark_name(name):
            res = name.split('/')[2]
            return res
        for pss in self.benchmark.pss_list:
            self.labels.append(get_benchmark_name(pss.name))

        self.matrix = None
        self.num_of_p = self.num_of_s = len(self.seqs)
        self.gen_matrix()

    def solve_one(self, pss, all_seq, sql_list):
        def trans(temp: ProSeqSiz, seq: list[int]) -> tuple:
            pro_seq2 = copy.deepcopy(temp)
            pro_seq2.sequence = seq
            return pro_seq2.export()

        env = EnvManager()
        size_list = []
        for seq in all_seq:
            tmp_pss = trans(pss, seq)
            size = env.evaluate_for_mp(tmp_pss, sql_list)
            size_list.append(size)

        return size_list

    def gen_matrix(self):

        args_list = [(pss, self.seqs) for pss in self.benchmark.pss_list]
        m = start_tasks(self.solve_one, args_list, desc='Coreset Gen')
        self.benchmark.cal_mean()
        self.matrix = np.array(m).T

    def gen_coreset(self, filepath, size=50, version=None):

        def cal(mt):
            def get_arith(size, oz) -> float:
                return 0.0 if math.isclose(oz, 0) else (oz - size) / oz

            def get_geo(size, oz):
                return 1.0 if math.isclose(oz, 0) else oz / size

            temp_dict, res_dict = {}, {}
            mn = np.min(mt, axis=0)
            for i in range(self.num_of_p):
                if self.labels[i] not in temp_dict:
                    temp_dict[self.labels[i]] = []
                temp_dict[self.labels[i]].append(get_arith(mn[i], self.benchmark.pss_list[i].get_Oz()))
            for k, v in temp_dict.items():
                res_dict[k] = get_arith_mean(v)
            return res_dict

        global_min = cal(self.matrix)

        def phi(bits: list[int], rate: float):
            if get_num_of_ones(bits) == 0:
                return False, None
            idxs = [index for index, value in enumerate(bits) if value == 1]
            now_dict = cal(self.matrix[idxs])
            ok = True
            for key in now_dict:
                # print(now_dict[key], global_min[key])
                if now_dict[key] < 0 or global_min[key] < 0:
                    if math.fabs(now_dict[key]) < math.fabs(global_min[key]) * rate:
                        ok = False
                else:
                    if now_dict[key] < global_min[key] * rate:
                        ok = False
            if ok:
                return ok, bits
            else:
                return ok, None

        def prob_dd(prob: [], rate):
            final_list = [1 for _ in range(self.num_of_s)]

            xt = [1] * self.num_of_s
            while True:
                # if all prob[i] equals to 1 or 0, then break
                if all(math.isclose(value, 1) or math.isclose(value, 0) for value in prob):
                    # print('prob:', prob)
                    return final_list

                choice = [(prob[i], i) for i in range(self.num_of_s) if xt[i] == 1]
                x = xt.copy()
                assert x == xt
                choice = sorted(choice)
                m = len(choice)

                expect, pass_pro = 0.0, 1.0
                for i in range(m):
                    p, idx = choice[i]
                    x[idx] = 0
                    pass_pro *= 1 - p
                    new_expect = (i + 1) * pass_pro
                    if expect > new_expect:
                        x[idx] = 1
                        break
                    else:
                        expect = new_expect

                # update probability
                t_or_f, new_list = phi(x, rate)
                if t_or_f:
                    final_list = new_list
                    prob = [0 if x[i] == 0 else prob[i] for i in range(self.num_of_s)]
                    xt = x.copy()
                else:
                    tmp = math.prod(1 - prob[i] for i in range(self.num_of_s) if x[i] == 0)
                    prob = [prob[i] / (1 - tmp) if x[i] == 0 else prob[i] for i in range(self.num_of_s)]

        def get_num_of_ones(bits):
            return sum([1 if e == 1 else 0 for e in bits])

        def check(rate, seed=0, _version=None):
            if _version == 'another version':
                rnd = random.Random()
                rnd.seed(seed)
                random_list = [rnd.uniform(0.3, 0.7) for _ in range(self.num_of_s)]
            else:
                random_list = [0.5 for _ in range(self.num_of_s)]
            bits = prob_dd(random_list, rate)
            num_of_ones = get_num_of_ones(bits)
            return bits if num_of_ones <= size else None

        print(f'Now, generate coreset')
        # binary search
        final_res_bits, final_res_rate = None, None
        for seed in range(1):
            low, high = 0, 1
            res_bits, res_rate = [1 for _ in range(self.num_of_s)], 0
            # for _ in range(10):
            for _ in range(20):
                mid_rate = (low + high) / 2
                temp_bits = check(mid_rate, seed)
                if temp_bits is not None:
                    if mid_rate >= res_rate:
                        res_bits, res_rate = temp_bits, mid_rate
                    low = mid_rate
                else:
                    high = mid_rate

            if final_res_rate is None or res_rate > final_res_rate:
                final_res_bits, final_res_rate = res_bits, res_rate
        print('best: ', final_res_rate, 'candidate: ', res_rate)

        if get_num_of_ones(final_res_bits) > size:
            final_res_bits = [0 for _ in range(self.num_of_s)]

        while get_num_of_ones(final_res_bits) < size:
            for i in range(len(final_res_bits)):
                if final_res_bits[i] == 0:
                    final_res_bits[i] = 1
                    break
        res = [self.seqs[i] for i in range(len(self.seqs)) if final_res_bits[i] == 1]

        save_coreset_to_file(res, filepath)
        print(f'Coreset generating done.')
        return Coreset(name='None', seq_list=res)
