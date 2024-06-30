import sys
sys.path.append('./../../')
import random
import statistics

from tqdm import tqdm

from Source.Data.ProSeqSiz import ProSeqSiz
from Source.DataBase.DBWrapper import DB
from Source.Env.EnvManager import EnvManager
from Source.Train.gen import load_train_pss
from Source.Util.parallel import start_tasks
from Source.Util.util import get_root_path, str_to_int_list, my_mkdir, unique, GLOBAL_VALUE


class Benchmark:
    """
    This class is a collection of ProSeqSiz.
    """
    def __init__(self, name: str, pss_list, k=998244353):
        """
        :param name:
        :param pss_list: if the type of pss_list is string, then call 'load_train_pss', else it's list of ProSeqSiz
        """
        self.name = name
        if isinstance(pss_list, list):
            self.pss_list = pss_list
        else:
            self.pss_list = load_train_pss(pss_list, k)
            self.sort_by_name()

    def cut(self, k=17500, seed=998244353):
        rnd = random.Random()
        rnd.seed(seed)
        rnd.shuffle(self.pss_list)
        self.pss_list = self.pss_list[:k]

    def sort_by_name(self):
        self.pss_list.sort(key=lambda pss: pss.name)

    def export_seq_list(self):
        self.sort_by_name()
        return [pss.sequence for pss in self.pss_list]

    def get_avg_len_of_seq_list(self):
        seq_list = self.export_seq_list()
        len_list = [len(seq) for seq in seq_list]
        return sum(len_list) / len(len_list)

    def sort_by_size(self, reverse=True):
        size_list = EnvManager.multi_evaluate([pss.export() for pss in self.pss_list])
        for idx, pss in enumerate(self.pss_list):
            pss.Size = size_list[idx]
        self.pss_list.sort(key=lambda pss: pss.get_size(), reverse=reverse)

    def merge_min(self, other):
        """
        During the iteration, ensure that the program select the optimal sequence from previous sequence.
        'self' and 'other' respectively represent the result of current iteration and previous iteration,
        combine the two and select the best.
        :param other:
        :return:
        """
        if other is None:
            return
        assert len(self.pss_list) == len(self.pss_list)
        self.sort_by_name()
        other.sort_by_name()
        for i in range(len(self.pss_list)):
            assert self.pss_list[i].name == other.pss_list[i].name
            self.pss_list[i] = min(self.pss_list[i], other.pss_list[i])

    def get_geo_mean(self):
        """
        Get the geometric mean of optimisation value
        :return:
        """
        self.cal_mean()
        values = [pss.get_geo() for pss in self.pss_list]
        return statistics.geometric_mean(values)

    def get_arith_mean(self):
        """
        Get the arith mean of optimisation value
        :return:
        """
        self.cal_mean()
        values = [pss.get_arith() for pss in self.pss_list]
        return statistics.mean(values)

    def get_num_of_seq(self) -> int:
        """
        Calculate different sequences among sequences corresponding to programs.
        function 'unique' in 'Source/Util/util.py' implements such a deduplication
        :return:
        """
        ls = unique([pss.sequence for pss in self.pss_list])
        return len(ls)

    def cal_Oz(self):
        # Oz
        tag = False
        for pss in self.pss_list:
            if pss.Oz == ProSeqSiz.INF:
                tag = True
        if tag:
            args_list = [((pss.name, pss.benchmark, [GLOBAL_VALUE.OZ]), ) for pss in self.pss_list]
            oz_list = start_tasks(EnvManager.evaluate_Oz, args_list, desc='Evaluate Oz')
            for i, pss in enumerate(self.pss_list):
                pss.Oz = oz_list[i]

    def cal_Size(self):
        # Size
        tag = False
        for pss in self.pss_list:
            if pss.Size == ProSeqSiz.INF:
                tag = True
        if tag:
            args_list = [((pss.name, pss.benchmark, pss.sequence),) for pss in self.pss_list]
            size_list = start_tasks(EnvManager.evaluate, args_list, desc='Evaluate Sequence')
            for i, pss in enumerate(self.pss_list):
                pss.Size = size_list[i]

    def cal_mean(self):
        self.cal_Oz()
        self.cal_Size()

    def cal_init(self):
        tag = False
        for pss in self.pss_list:
            if pss.Init == ProSeqSiz.INF:
                tag = True
        if tag:
            args_list = [((pss.name, pss.benchmark, [GLOBAL_VALUE.INIT]), ) for pss in self.pss_list]
            init_list = start_tasks(EnvManager.evaluate_init, args_list, desc='Evaluate Init Size')
            for i, pss in enumerate(self.pss_list):
                pss.Init = init_list[i]

    def save_self(self, file_path):
        self.sort_by_name()
        self.cal_mean()
        self.cal_init()
        with open(file_path, 'w') as f:
            f.write(f'total {len(self.pss_list)} programs and {self.get_num_of_seq()} different sequences, '
                    f'geo_mean={"{:.5f}".format(self.get_geo_mean())}, '
                    f'arith_mean={"{:.3%}".format(self.get_arith_mean())}\n')
            for pss in self.pss_list:
                f.write(f'{pss.name} ~ {pss.sequence}'
                        f' ~ init-ir-count:{"{:.3f}".format(pss.get_init())}'
                        f' ~ geo:{"{:.3f}".format(pss.get_geo())}'
                        f' ~ arith:{"{:.1%}".format(pss.get_arith())}\n')

    def save(self, algo_name: str, iteration: int, record=True, file_path=None):
        """
        Log information into local file
        """
        if file_path is not None:
            self.save_self(file_path)
        else:
            dir_path = f'{get_root_path()}/Result/{algo_name}/{self.name}/'
            my_mkdir(dir_path)
            filepath = f'{dir_path}/iter_{iteration}.train_result'
            self.save_self(filepath)

        if record:
            # high_view = f'{dir_path}/__high_view.txt'
            # mode = 'a' if iteration > 0 else 'w'
            # with open(high_view, mode) as f:
            #     f.write(f'After {iteration} iterations, '
            #             f'total {len(self.pss_list)} programs and {self.get_num_of_seq()} different sequences, '
            #             f'geo_mean={"{:.5f}".format(self.get_geo_mean())}, '
            #             f'arith_mean={"{:.3%}".format(self.get_arith_mean())}\n')
            pass

    def export_seqs(self, algo_name: str, iteration: int):
        """
        Save all pass sequences corresponding to programs into local file
        """
        dir_path = f'{get_root_path()}/Result/{algo_name}/{self.name}/'
        with open(f'{dir_path}/iter_{iteration}_myset.txt', 'w') as f:
            for pss in self.pss_list:
                f.write(f'{pss.sequence}\n')
        # dir_path = f'{get_root_path()}/Result/{algo_name}/{self.name}/'
        # with open(f'{dir_path}/iter_{iteration}_myset_with_benchmark.txt', 'w') as f:
        #     for pss in self.pss_list:
        #         f.write(f'{pss.sequence}&{pss.name}\n')

    def load_from_db(self, table_name: str) -> bool:
        """
        Load benchmark from database if this iteration has been finished before.
        This can avoid redundant computations.
        """
        if not DB.db.exist_table(table_name):
            return False
        name_benchmark = {}
        for pss in self.pss_list:
            name_benchmark[pss.name] = pss.benchmark
        res = DB.db.select_all(table_name)
        if res is None:
            return False
        else:
            self.pss_list = []
            for name, str_sequence, size in res:
                self.pss_list.append(ProSeqSiz(name, name_benchmark.get(name, None),
                                               str_to_int_list(str_sequence), size))
            return True

    def save_to_db(self, table_name: str) -> bool:
        """
        Inverse process of 'load_from_db'
        """
        DB.db.delete_table_if_exist(table_name)
        DB.db.create_table(table_name,
                                   ['program', 'sequence', 'size'],
                                   ['TEXT', 'TEXT', 'INT'],
                                   ['program', 'sequence'])

        for pss in self.pss_list:
            # print('saving...')
            DB.db.insert(table_name, [pss.name, pss.sequence, pss.get_size()])
        return True

    def debug(self):
        print(f'=============[benchmark {self.name}]==============')
        print(f'')
        print(f'pss_list:')
        for pss in self.pss_list:
            print(f'{pss.name}, {pss.sequence}, {pss.get_size()}')
        print(f'==================================================')
