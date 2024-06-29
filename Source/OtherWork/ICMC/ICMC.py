import copy

from Source.Data.Benchmark import Benchmark
from Source.Data.BoostPassSet import BoostPassSet
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import EvalCoreset
from Source.Evaluation.model_evaluation import ModelEvaluation
from Source.OtherWork.ICMC.Subseq import SubSeqGen
from Source.Solver.CoresetGen.CoresetGen import CoresetGen
from Source.Util.parallel import start_tasks
from Source.Util.util import my_mkdir, get_geo_mean, get_arith_mean, save_coreset_to_file, get_root_path
import argparse


class ICMC:
    def __init__(self, ssg: SubSeqGen, benchmark: Benchmark):
        self.boost_passes = BoostPassSet(ssg.subseq_set)
        self.benchmark = benchmark
        self.ssg = ssg
        pass

    def get_ex(self, ls1: list, ls2: list) -> list:
        n = len(ls2)
        assert ls1[:n] == ls2[:n]
        return ls1[n:]

    def select_main(self, pro_seq_siz: ProSeqSiz, env, sql_list, eval_limits):
        def trans(pss: ProSeqSiz, seq: list[int]) -> tuple:
            pro_seq2 = copy.deepcopy(pss)
            pro_seq2.extend(seq)
            return pro_seq2.export()

        pss_list = []
        for main in self.boost_passes:
            if eval_limits > 0:
                eval_limits -= 1
                pss_list.append(trans(pro_seq_siz, main))
        size_list = env.multi_evaluate_for_mp(pss_list, sql_list)
        result_pss_list = [ProSeqSiz.import_pss(pss, size) for pss, size in zip(pss_list, size_list)]
        result = min(result_pss_list)

        return result, self.get_ex(result.sequence, pro_seq_siz.sequence), eval_limits

    def solve_one(self, pro_seq_siz: ProSeqSiz, sql_list):
        # print(f'Now, dealing with {pro_seq_siz.name} ...')
        env = EnvManager()
        res = copy.deepcopy(pro_seq_siz)
        res.sequence = []

        eval_limits = 50000
        for _ in range(12):
            new_res, main, eval_limits = self.select_main(res, env, sql_list, eval_limits)

            if new_res < res:
                res = new_res
            else:
                break

        return res

    def solve(self):
        self.boost_passes.append([])

        args = [(pss, ) for pss in self.benchmark.pss_list]
        res = start_tasks(self.solve_one, args, desc=f'ICMC K:{self.ssg.K} c:{self.ssg.c}')
        self.benchmark.pss_list = res
        self.evaluation()

    def evaluation(self):
        my_mkdir(f'eval')
        prefix = f'{get_root_path()}/OtherWork/ICMC'
        save_file_name = f'{prefix}/eval/eval_{self.ssg.get_identify()}.txt'
        dataset = {}
        for pss in self.benchmark.pss_list:
            name = ModelEvaluation.get_benchmark_name(pss.name)
            if name not in dataset:
                dataset[name] = []
            dataset[name].append(pss)

        geo_ls, arith_ls = [], []
        with open(save_file_name, 'w') as _:
            pass
        for dataset_name, _pss_list in dataset.items():
            benchmark = Benchmark(dataset_name, _pss_list)
            with open(save_file_name, 'a') as f:
                geo, arith = benchmark.get_geo_mean(), benchmark.get_arith_mean()
                f.write(f'benchmark={benchmark.name}, '
                        f'{"{:.3f}".format(geo)}/'
                        f'{"{:.1%}".format(arith)}\n')
                geo_ls.append(geo)
                arith_ls.append(arith)
        with open(save_file_name, 'a') as f:
            f.write(f'geo_mean={"{:.3f}".format(get_geo_mean(geo_ls))} '
                    f'arith_mean={"{:.1%}".format(get_arith_mean(arith_ls))}'
                    f'\n')

        self.benchmark.save(None, None, record=False, file_path=f'{prefix}/icmc_{self.ssg.get_identify()}.txt')

        temp_benchmark = copy.deepcopy(self.benchmark)
        coreset_gen = CoresetGen(temp_benchmark)
        coreset = coreset_gen.gen_coreset(f'{prefix}/icmc_{self.ssg.get_identify()}_coreset.txt')
        save_coreset_to_file(coreset, f'{prefix}/icmc_{self.ssg.get_identify()}_coreset.txt')


def parse():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument('-k', '--k', required=True, type=int, help='Number of pass')
    parser.add_argument('-c', '--c', required=True, type=int, help='Number of sequence')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()

    # Train
    train_benchmark = Benchmark('__it_train', '__it_train')

    # for num_of_kept_pass in [120, 110, 100, 90, 80, 70]:
    #     for num_of_subseq in [8, 7, 6, 5, 4]:
    num_of_kept_pass, num_of_subseq = args.k, args.c
    num_of_pass = 124

    ssg = SubSeqGen(
        f'__it_train',
        O=range(0, num_of_pass),
        P=train_benchmark.pss_list,
        K=num_of_kept_pass,
        c=num_of_subseq,
    )

    if not ssg.load():
        ssg.gen()
    ssg.save()

    # Test
    test_benchmark = Benchmark('__it_train', '__it_train')
    icmc = ICMC(ssg, test_benchmark)
    icmc.solve()
