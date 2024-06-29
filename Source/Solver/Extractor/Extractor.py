import copy
import os

from Source.Data.Benchmark import Benchmark
from Source.Data.BoostPassSet import BoostPassSet
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import GLOBAL_VALUE


class Extractor:
    def __init__(self):
        self.max_concurrency = os.cpu_count()
        pass

    def extract(self, pss: ProSeqSiz, sql_list):
        env = EnvManager()
        solved_tasks = [env.evaluate_whole_for_mp(pss.export(), sql_list)]
        for i in range(len(pss.sequence)):
            temp = copy.deepcopy(pss)
            temp.sequence = pss.sequence[:i] + pss.sequence[i + 1:]
            solved_tasks.append(env.evaluate_whole_for_mp(temp.export(), sql_list))

        def check(low: int, high: int):
            keep_base = solved_tasks[0][low]  # [0:low - 1]
            keep_final = solved_tasks[0][high + 1]  # [low: high]
            remove_base = solved_tasks[low][low - 1]  # [0: low - 2]
            remove_final = solved_tasks[low][high - 1 + 1]  # [low: high]

            if remove_base - remove_final < keep_base - keep_final:
                return True
            else:
                return False

        # extract subsequence
        action_set = []
        n = len(pss.sequence)
        i = n - 1
        while i >= 0:
            j = i
            while j > 0 and check(j, i):
                j = j - 1
            action_set.append(pss.sequence[j: i + 1])
            i = j - 1

        return BoostPassSet(action_set)

    def solve(self, benchmark: Benchmark, table_name=None, cal=GLOBAL_VALUE.RAY) -> BoostPassSet:
        print(f'Now, extract BPC from pass sequence.')

        boost_pass_set = BoostPassSet([])
        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            if boost_pass_set.load_from_db(table_name):
                print(f'[Load from db] table name: {table_name}')
                return boost_pass_set

        benchmark.sort_by_size()
        args = [(pss, ) for pss in benchmark.pss_list]
        res = start_tasks(self.extract, args, desc='Extract')
        for bs in res:
            boost_pass_set += bs

        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            print(f'[Save to db] table name: {table_name}')
            boost_pass_set.save_to_db(table_name)
        return boost_pass_set
