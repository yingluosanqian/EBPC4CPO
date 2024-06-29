import copy

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import get_geo_mean, get_arith_mean, get_benchmark_name, my_mkdir, GLOBAL_VALUE, get_root_path


class Trans:
    @staticmethod
    def copy_and_reset(pss: ProSeqSiz, seq: list[int]) -> tuple:
        pro_seq2 = copy.deepcopy(pss)
        pro_seq2.sequence = seq
        return pro_seq2.export()


class EvalCoreset:
    @staticmethod
    def solve_one(pss, coreset, sql_list):
        env = EnvManager()
        res = []
        for seq in coreset:
            new_pss = Trans.copy_and_reset(pss, seq)
            size = env.evaluate_for_mp(new_pss, sql_list)
            res.append(ProSeqSiz.import_pss(new_pss, size))
            # res.append(ProSeqSiz.import_pss(*(env.evaluate_for_mp_with_cut(new_pss, sql_list))))
        return min(res)

    @staticmethod
    def evaluate(test_benchmark: Benchmark, coreset: Coreset, filepath=None):
        args = [(pss, coreset) for pss in test_benchmark.pss_list]
        test_benchmark.pss_list = start_tasks(EvalCoreset.solve_one, args, desc='Eval Coreset')
        if filepath is None:
            filepath = f'{get_root_path()}/Evaluation/result.txt'
        dataset = {}
        for pss in test_benchmark.pss_list:
            name = get_benchmark_name(pss.name)
            if name not in dataset:
                dataset[name] = []
            dataset[name].append(pss)

        geo_ls, arith_ls = [], []
        with open(filepath, 'w') as _:
            pass
        for dataset_name in GLOBAL_VALUE.TABLE_ORDER:
            if dataset_name in dataset:
                _pss_list = dataset[dataset_name]
                benchmark = Benchmark(dataset_name, _pss_list)
                with open(filepath, 'a') as f:
                    geo, arith = benchmark.get_geo_mean(), benchmark.get_arith_mean()
                    f.write(f'benchmark={benchmark.name}, '
                            f'{"{:.3f}".format(geo)}/'
                            f'{"{:.1%}".format(arith)}\n')
                    geo_ls.append(geo)
                    arith_ls.append(arith)
        with open(filepath, 'a') as f:
            f.write(f'geo_mean={"{:.3f}".format(get_geo_mean(geo_ls))} '
                    f'arith_mean={"{:.1%}".format(get_arith_mean(arith_ls))}'
                    f'\n')
        test_benchmark.save(None, None, record=False, file_path=f'{filepath}_detail.txt')


def gen_latex_table():

    file_path = f'{get_root_path()}/Evaluation/coreset/eval/eval_coreset.txt'
    with open(file_path, 'w') as f:
        f.write('')

    def float_to_str_arith(value):
        return f'{value: .1f}\%'
        # return f'{value: .3f}'

    def bold_arith(value):
        return '\\textbf{' + float_to_str_arith(value) + '}'

    # arith
    res = []
    for set_file, _ in [
        ('bpc_coreset', 'bpc_coreset'),
        ('nvp_coreset_1', 'nvp_coreset_1.txt'),
        ('nvp_coreset_2', 'nvp_coreset_2.txt'),
        ('icmc_coreset', 'icmc_coreset.txt'),
        ('gens_coreset', 'gens_coreset.txt'),
    ]:
        temp_res = [0] * len(GLOBAL_VALUE.TABLE_ORDER)
        with open(f'{get_root_path()}/Evaluation/coreset/eval/results_{set_file}.txt', 'r') as f:
            for line in f:
                if 'benchmark' in line:
                    arith = float(line.split(' ')[1].split('/')[1].replace('%', ''))
                    # geo = float(line.split(' ')[1].split('/')[0])
                    dataset_name = line.split(',')[0].split('=')[1]
                    idx = GLOBAL_VALUE.TABLE_ORDER.index(dataset_name)
                    temp_res[idx] = arith
        res.append(temp_res)

    print(res)

    m, n = len(res), len(res[0])
    with open(file_path, 'w') as f:
        for i in range(n):
            sentence = f'{GLOBAL_VALUE.TABLE_ORDER[i]}'
            values = [res[j][i] for j in range(m)]
            max_value = max(values)
            values_str = [bold_arith(values[j]) if values[j] == max_value else float_to_str_arith(values[j]) for j in range(m)]
            for value_str in values_str:
                sentence += f' & {value_str}'
            sentence += ' \\\\'
            f.write(sentence)
            print(sentence)


    def float_to_str_geo(value):
        # return f'{value: .1f}\%'
        return f'{value: .3f}'

    def bold_geo(value):
        return '\\textbf{' + float_to_str_geo(value) + '}'

    # geo mean
    res = []
    for set_file, _ in [
        ('bpc_coreset', 'bpc_coreset'),
        ('nvp_coreset_1', 'nvp_coreset_1.txt'),
        ('nvp_coreset_2', 'nvp_coreset_2.txt'),
        ('icmc_coreset', 'icmc_coreset.txt'),
        ('gens_coreset', 'gens_coreset.txt'),
    ]:
        temp_res = [0] * len(GLOBAL_VALUE.TABLE_ORDER)
        with open(f'{get_root_path()}/Evaluation/coreset/eval/results_{set_file}.txt', 'r') as f:
            for line in f:
                if 'benchmark' in line:
                    # arith = float(line.split(' ')[1].split('/')[1].replace('%', ''))
                    geo = float(line.split(' ')[1].split('/')[0])
                    dataset_name = line.split(',')[0].split('=')[1]
                    idx = GLOBAL_VALUE.TABLE_ORDER.index(dataset_name)
                    temp_res[idx] = geo
        res.append(temp_res)

    print(res)

    m, n = len(res), len(res[0])
    with open(file_path, 'w') as f:
        for i in range(n):
            sentence = f'{GLOBAL_VALUE.TABLE_ORDER[i]}'
            values = [res[j][i] for j in range(m)]
            max_value = max(values)
            values_str = [bold_geo(values[j]) if values[j] == max_value else float_to_str_geo(values[j]) for j in range(m)]
            for value_str in values_str:
                sentence += f' & {value_str}'
            sentence += ' \\\\'
            print(sentence)
            f.write(sentence+'\n')


def eval_all():

    my_mkdir('coreset/eval')
    for set_file, set_name in [
        ('bpc_coreset', 'bpc_coreset.txt'),
        ('nvp_coreset_1', 'nvp_coreset_1.txt'),
        ('nvp_coreset_2', 'nvp_coreset_2.txt'),
        ('icmc_coreset', 'icmc_coreset.txt'),
        ('gens_coreset', 'gens_coreset.txt'),
    ]:
        coreset = Coreset(f'{set_file}')
        EvalCoreset.evaluate(Benchmark('__test', '__test'), coreset, filepath=f'coreset/eval/results_{set_name}')


if __name__ == '__main__':
    eval_all()
    gen_latex_table()
