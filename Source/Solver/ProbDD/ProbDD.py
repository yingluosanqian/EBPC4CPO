import copy
import math
import os

from Source.Data.Benchmark import Benchmark
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import GLOBAL_VALUE


class ProbDD:
    def __init__(self):
        self.max_concurrency = os.cpu_count()
        pass

    def prob_dd(self, pss: ProSeqSiz, sql_list):
        def phi(ps: ProSeqSiz, bitset: list[int]) -> (bool, ProSeqSiz):
            ps2 = copy.deepcopy(ps)
            ps2.sequence = [e1 for e1, e2 in zip(ps2.sequence, bitset) if e2 == 1]
            ps2 = ProSeqSiz.import_pss(ps2.export(), env.evaluate_for_mp(ps2.export(), sql_list))
            return ps2 <= ps, ps2

        env = EnvManager()
        result = copy.deepcopy(pss)
        n = len(pss.sequence)
        prob = [0.5 for _ in range(n)]

        xt = [1] * n
        while True:
            # if all prob[i] equals to 1 or 0, then break
            if all(math.isclose(value, 1) or math.isclose(value, 0) for value in prob):
                # print('prob:', prob)
                return result

            choice = [(prob[i], i) for i in range(n) if xt[i] == 1]
            x = xt.copy()
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
            t_or_f, temp = phi(pss, x)
            if t_or_f:
                if temp <= result:
                    result = temp
                prob = [0 if x[i] == 0 else prob[i] for i in range(n)]
                xt = x.copy()
            else:
                tmp = math.prod(1 - prob[i] for i in range(n) if x[i] == 0)
                prob = [prob[i] / (1 - tmp) if x[i] == 0 else prob[i] for i in range(n)]

    def solve(self, benchmark: Benchmark, table_name=None, cal=GLOBAL_VALUE.RAY):
        print(f'Now, refine pass sequence: trim pass sequence.')

        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            if benchmark.load_from_db(table_name):
                print(f'[Load from db] table name: {table_name}')
                return

        # Solve one by one
        benchmark.sort_by_size()
        args = [(pss, ) for pss in benchmark.pss_list]
        res = start_tasks(self.prob_dd, args, desc='Trim')
        benchmark.pss_list = res

        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            print(f'[Save to db] table name: {table_name}')
            benchmark.save_to_db(table_name)
