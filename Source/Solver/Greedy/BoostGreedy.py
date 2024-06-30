import sys
sys.path.append('./../../../')
import copy
from Source.Data.Benchmark import Benchmark
from Source.Data.BoostPassSet import BoostPassSet
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import GLOBAL_VALUE


class BoostGreedy:
    def __init__(self):
        pass

    def get_ex(self, ls1: list, ls2: list) -> list:
        n = len(ls2)
        assert ls1[:n] == ls2[:n]
        return ls1[n:]

    def select_pre(self, pro_seq_siz: ProSeqSiz, boost_passes: BoostPassSet, main: list[int], env, sql_list):
        def trans(pss: ProSeqSiz, seq1: list[int], seq2: list[int]) -> tuple:
            pro_seq2 = copy.deepcopy(pss)
            pro_seq2.extend(seq1)
            pro_seq2.extend(seq2)
            return pro_seq2.export()

        pss_list = [trans(pro_seq_siz, pre, main) for pre in boost_passes]
        size_list = env.multi_evaluate_for_mp(pss_list, sql_list)
        result_pss_list = [ProSeqSiz.import_pss(pss, size) for pss, size in zip(pss_list, size_list)]
        result = min(result_pss_list)

        return result, self.get_ex(result.sequence, pro_seq_siz.sequence)

    def select_main(self, pro_seq_siz: ProSeqSiz, boost_passes: BoostPassSet, env, sql_list):
        def trans(pss: ProSeqSiz, seq: list[int]) -> tuple:
            pro_seq2 = copy.deepcopy(pss)
            pro_seq2.extend(seq)
            return pro_seq2.export()

        pss_list = [trans(pro_seq_siz, main) for main in boost_passes]
        size_list = env.multi_evaluate_for_mp(pss_list, sql_list)
        result_pss_list = [ProSeqSiz.import_pss(pss, size) for pss, size in zip(pss_list, size_list)]
        result = min(result_pss_list)

        return self.get_ex(result.sequence, pro_seq_siz.sequence)

    def solve_one(self, pro_seq_siz: ProSeqSiz, boost_passes: BoostPassSet, sql_list):
        # print(f'Now, dealing with {pro_seq_siz.name} ...')
        env = EnvManager()
        res = copy.deepcopy(pro_seq_siz)
        res.sequence = []

        step_count = 12
        for _ in range(step_count):
            main = self.select_main(res, boost_passes, env, sql_list)
            new_res, pre_main = self.select_pre(res, boost_passes, main, env, sql_list)

            if new_res < res:
                res = new_res
            else:
                break

        return res

    def solve(self, benchmark: Benchmark, boost_passes: BoostPassSet, table_name=None, cal=GLOBAL_VALUE.RAY):
        print(f'Now, build pass sequence for each program.')
        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            if benchmark.load_from_db(table_name):
                # benchmark.debug()  # debug
                print(f'[Load from db] table name: {table_name}')
                return

        cp = copy.deepcopy(boost_passes)
        cp.seq_set.append([])

        benchmark.sort_by_size()
        args = [(pss, cp) for pss in benchmark.pss_list]
        res = start_tasks(self.solve_one, args, desc='Forward-Backward Greedy', order=False)
        benchmark.pss_list = res

        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            print(f'[Save to db] table name: {table_name}')
            benchmark.save_to_db(table_name)
