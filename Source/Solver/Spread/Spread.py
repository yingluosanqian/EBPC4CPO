import copy
import os


from Source.Data.Benchmark import Benchmark
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import unique, GLOBAL_VALUE


class Spread:

    def __init__(self):
        self.max_concurrency = os.cpu_count()
        pass

    def solve_one(self, pss: ProSeqSiz, all_seq, sql_list):
        def trans(pss_tmp: ProSeqSiz, seq_tmp: list[int]) -> tuple:
            pro_seq2 = copy.deepcopy(pss_tmp)
            pro_seq2.sequence = seq_tmp
            return pro_seq2.export()

        env = EnvManager()
        pss_list = []
        for seq in all_seq:
            tmp = trans(pss, seq)
            size = env.evaluate_for_mp(tmp, sql_list)
            pss_list.append(ProSeqSiz.import_pss(tmp, size))

        return min(pss_list)

    def solve(self, benchmark: Benchmark, table_name: str, cal=GLOBAL_VALUE.RAY):
        print(f'Now, refine pass sequence: remove suboptimal pass sequence.')
        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            if benchmark.load_from_db(table_name):
                print(f'[Load from db] table name: {table_name}')
                return

        benchmark.sort_by_size()
        all_seq = unique([ps.sequence for ps in benchmark.pss_list])
        args = [(pss, all_seq) for pss in benchmark.pss_list]
        res = start_tasks(self.solve_one, args, desc='Removal')
        benchmark.pss_list = res

        # Cache Cache Cache, I LOVE CACHE
        if table_name is not None:
            print(f'[Save to db] table name: {table_name}')
            benchmark.save_to_db(table_name)


if __name__ == '__main__':
    pass
