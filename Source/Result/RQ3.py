import sys

sys.path.append('./../../')
import copy
import random
from pathlib import Path


import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
import numpy as np

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import EvalCoreset, Trans
from Source.Solver.CoresetGen.CoresetGen import CoresetGen
from Source.Util.parallel import start_tasks
from Source.Util.util import read_coreset_to_file, get_benchmark_name, GLOBAL_VALUE, get_geo_mean, get_arith_mean, \
    get_geo, get_arith
from Source.Util.util import get_root_path
import argparse

ITERATIONS = 3


def load_50_result(method, metric):
    path = Path(f'{get_root_path()}') / f'Result' / Path(f'{method}') / f'__it_train'
    ls = []
    for i in range(ITERATIONS):
        file = path / f'__test_result_{i}.txt'
        with file.open('r') as f:
            for line in f:
                if 'geo_mean' in line:
                    if metric == 'geo_mean':
                        line = line.strip().split(' ')[0]
                        value = float(line.split('=')[1])
                        ls.append(value)
                    elif metric == 'arith_mean':
                        line = line.replace('%', '').strip().split(' ')[1]
                        value = float(line.split('=')[1])
                        ls.append(value)
    return ls


def abla_1():
    data_base = load_50_result(f'base', f'geo_mean')
    data_no_trimming = load_50_result(f'no_trimming', f'geo_mean')
    data_no_sharing = load_50_result(f'no_removal', f'geo_mean')
    print('base (GMeanOverOz):', data_base[ITERATIONS - 1])
    print('-trimming (GMeanOverOz):', data_no_trimming[ITERATIONS - 1])
    print('-removal (GMeanOverOz):', data_no_sharing[ITERATIONS - 1])


def solve_all_res(pss, coreset, sql_list):
    env = EnvManager()
    res = []
    for seq in coreset:
        new_pss = Trans.copy_and_reset(pss, seq)
        size = env.evaluate_for_mp(new_pss, sql_list)
        res.append(size)
    return res


def my_eval(temp_res, test_benchmark, dataset_benchmark: {}, filepath):
    dataset = {}
    for size, pss in zip(temp_res, test_benchmark.pss_list):
        name = get_benchmark_name(pss.name)
        if name not in dataset:
            dataset[name] = []
        dataset[name].append(size)

    with open(filepath, 'w') as f:
        pass
    geo_ls, arith_ls = [], []
    for dataset_name in GLOBAL_VALUE.TABLE_ORDER:
        if dataset_name in dataset:
            tar_benchmark = dataset_benchmark[dataset_name]
            for i in range(len(dataset[dataset_name])):
                tar_benchmark.pss_list[i].Size = dataset[dataset_name][i]
            with open(filepath, 'a') as f:
                geo = get_geo_mean([get_geo(pss.Size, pss.Oz) for pss in tar_benchmark.pss_list])
                arith = get_arith_mean([get_arith(pss.Size, pss.Oz) for pss in tar_benchmark.pss_list])
                f.write(f'benchmark={tar_benchmark.name}, '
                        f'{"{:.3f}".format(geo)}/'
                        f'{"{:.1%}".format(arith)}\n')
                geo_ls.append(geo)
                arith_ls.append(arith)
    with open(filepath, 'a') as f:
        f.write(f'geo_mean={"{:.3f}".format(get_geo_mean(geo_ls))} '
                f'arith_mean={"{:.1%}".format(get_arith_mean(arith_ls))}'
                f'\n')


def abla_2_1(iter):
    seqs = read_coreset_to_file(f'base/__it_train/iter_{iter-1}_myset.txt')
    train_benchmark = Benchmark('__it_train', '__it_train')
    test_benchmark = Benchmark('__test', '__test')
    coreset_path = Path('abla_coreset_gen')
    coreset_path.mkdir(parents=True, exist_ok=True)
    coreset_gen = CoresetGen(copy.deepcopy(train_benchmark), seqs=seqs)

    args = [(pss, seqs) for pss in test_benchmark.pss_list]
    res = start_tasks(solve_all_res, args, desc='Eval All Seq')
    res = np.array(res).T
    print('Shape of res: ', res.shape)

    # process dataset_benchmark
    dataset, dataset_benchmark = {}, {}
    for pss in test_benchmark.pss_list:
        name = get_benchmark_name(pss.name)
        if name not in dataset:
            dataset[name] = []
        dataset[name].append(pss)
    for dataset_name in GLOBAL_VALUE.TABLE_ORDER:
        if dataset_name in dataset:
            _pss_list = dataset[dataset_name]
            dataset_benchmark[dataset_name] = Benchmark(dataset_name, _pss_list)
            dataset_benchmark[dataset_name].cal_Oz()
    random.seed(998244353)
    np.random.seed(998244353)
    for i in range(150):
        size = i + 1
        indices = list(np.random.choice(np.arange(0, len(seqs)), size))
        temp_res = res[indices].min(axis=0)
        my_eval(temp_res, test_benchmark, dataset_benchmark, coreset_path / f'result_coreset_{size}_random.txt')
    for i in range(150):
        size = i + 1
        coreset = coreset_gen.gen_coreset(coreset_path / f'coreset_{size}', size=size)
        indices = [seqs.index(seq) for seq in coreset]
        temp_res = res[indices].min(axis=0)
        my_eval(temp_res, test_benchmark, dataset_benchmark, coreset_path / f'result_coreset_{size}.txt')

def abla_2_2():
    N = 150
    coreset_path = Path('abla_coreset_gen')
    geo_ls, random_ls = [], []
    for i in range(N):
        size = i + 1
        result_file = coreset_path / f'result_coreset_{size}.txt'
        with result_file.open('r') as f:
            for line in f:
                if 'geo_mean' in line:
                    geo_ls.append(float(line.split(' ')[0].split('=')[1]) / 1.111)
        result_file = coreset_path / f'result_coreset_{size}_random.txt'
        with result_file.open('r') as f:
            for line in f:
                if 'geo_mean' in line:
                    random_ls.append(float(line.split(' ')[0].split('=')[1]) / 1.111)

    data_bs = np.array(geo_ls) * 100  # Convert to percentage
    data_bs2 = np.array(random_ls) * 100  # Convert to percentage

    # Set up the x-axis values (iterations 1 to 150)
    iterations = np.arange(1, N + 1)

    # Create the plot
    plt.figure(figsize=(12, 10))
    plt.plot(iterations, data_bs, label='CoresetGen (ours)', linestyle='-', linewidth=3)
    plt.plot(iterations, data_bs2, label='Random Policy', linestyle='none', marker='.', markersize=12, color='green')

    # Adding labels and title
    plt.xlabel('Size of coreset', fontsize=30)
    plt.ylabel('Code Size Reduction Effectiveness (%)', fontsize=30)

    # Mark special points
    special_points = [10, 50, 120]
    for point in special_points:
        plt.plot(point, data_bs[point-1], marker='s', markersize=30, markerfacecolor='none', color='red')  # Mark the point with a red circle
        plt.text(point, data_bs[point-1] + 2, f'({point}, {data_bs[point-1]:.2f}%)',
                 fontsize=25, ha='left', va='bottom')

    # special_points = [10, 50, 120]
    # for point in special_points:
    #     plt.plot(point, data_bs2[point - 1], marker='*', markersize=20, markerfacecolor='none', color='red')  # Mark the point with a red circle
    #     plt.text(point, data_bs2[point - 1], f'({point}, {data_bs2[point - 1]:.2f}%)',
    #              fontsize=20, ha='left', va='bottom')

    # Customize ticks and grid
    plt.xticks(np.arange(0, N+1, 20), fontsize=30)
    plt.yticks(fontsize=30)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Adding legend
    plt.legend(loc='lower right', fontsize=25, frameon=True, shadow=True, borderpad=1)

    # Display the plot
    plt.savefig('ablation_2.pdf', format='pdf', bbox_inches='tight')
    plt.show()


def parse():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument('-i', '--iter', required=True, type=int, help='Number of iterations')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    abla_1()
    abla_2_1(args.iter)
    abla_2_2()
    pass
