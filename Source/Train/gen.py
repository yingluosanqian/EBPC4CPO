import sys
sys.path.append('./../../../')
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import gym
from tqdm import tqdm

from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import get_root_path, get_geo_mean


def gen_all():
    # Train Val Test
    online_benchmarks = [
        ('anghabench-v1', 7000, 1000, 2000),
        ('blas-v0', 100, 100, 100),
        ('github-v0', 7000, 1000, 1000),
        ('linux-v0', 5000, 1000, 1000),
        ('opencv-v0', 100, 100, 100),
        ('poj104-v1', 7000, 1000, 1000),
        ('tensorflow-v0', 500, 100, 100),
        ('clgen-v0', 300, 200, 200),
        ('cbench-v1', 0, 0, 22),  # unseen
        ('chstone-v0', 0, 0, 12),  # unseen
        ('mibench-v1', 0, 0, 40),  # unseen
        ('npb-v0',  0, 0, 122),  # unseen
    ]
    online_generator = [
        ('csmith-v0', 200, 100, 100),
        ('llvm-stress-v0', 700, 200, 200),
    ]
    local_benchmarks = [
        ('codeforces', 0, 0, 300),
        ('atcoder', 0, 0, 300),
    ]

    train_file = f'Benchmark/__train.txt'
    search_train_file = f'Benchmark/search_train.txt'
    it_train_file = f'Benchmark/__it_train.txt'
    val_file = f'Benchmark/__val.txt'
    test_file = f'Benchmark/__test.txt'
    with open(train_file, 'w') as f:
        pass
    with open(it_train_file, 'w') as f:
        pass
    with open(val_file, 'w') as f:
        pass
    with open(test_file, 'w') as f:
        pass

    bad_program = {
        'None',
    }
    with open(f'BadBenchmark/bad.txt') as f:
        for line in f:
            line = line.strip()
            if line != '':
                if 'Online' in line or 'Local' in line:
                    line = line.split(' ')[1]
                bad_program.add(line)
    large_program = {
        'None',
    }
    with open(f'BadBenchmark/large_graph_benchmark.txt') as f:
        for line in f:
            line = line.strip()
            if line != '':
                if 'Online' in line or 'Local' in line:
                    line = line.split(' ')[1]
                large_program.add(line)
    print(f'bad benchmark: {bad_program}')
    nums_per_benchmark = 50
    train_list = []

    def is_bad_benchmark(benchmark_uri):
        for e in bad_program:
            if re.match(e, benchmark_uri):
                return True
        return False

    def append(ls, filepath, pre):
        with open(filepath, 'a') as fp:
            for benchmark_uri in ls:
                if not is_bad_benchmark(benchmark_uri):
                    fp.write(f'{pre} {benchmark_uri}\n')

    def reset(ls, filepath, pre):
        with open(filepath, 'w') as fp:
            for benchmark_uri in ls:
                if not is_bad_benchmark(benchmark_uri):
                    fp.write(f'{pre} {benchmark_uri}\n')

    def test_reset(ls, filepath, pre):
        # with open(filepath, 'w') as fp:
        #     for benchmark_uri in ls:
        #         if not is_bad_benchmark(benchmark_uri) and benchmark_uri not in large_program:
        #             fp.write(f'{pre} {benchmark_uri}\n')
        with open(test_file, 'a') as fp:
            for benchmark_uri in ls:
                if not is_bad_benchmark(benchmark_uri) and benchmark_uri not in large_program:
                    fp.write(f'{pre} {benchmark_uri}\n')

    with gym.make("llvm-ic-v0") as env:
        for name, train, val, test in online_benchmarks:
            all_programs = [uri for uri in env.datasets[name].benchmark_uris()]
            print(f'{name}: {len(all_programs)}')
            sampled_programs = random.sample(all_programs, min(len(all_programs), train + val + test))
            append(sampled_programs[:train], train_file, 'Online')
            append(sampled_programs[:min(train, nums_per_benchmark)], it_train_file, 'Online')
            reset(sampled_programs[:min(train, nums_per_benchmark)], f'Benchmark/__train_{name}.txt', 'Online')
            tmp = name.split('-')[0]
            train_list.append(f'{tmp} __train_{name}')
            append(sampled_programs[train: train + val], val_file, 'Online')
            if test > 0:
                test_reset(sampled_programs[train + val: train + val + test], f'Benchmark/final_test_{name}.txt', 'Online')

        for name, train, val, test in online_generator:
            sampled_programs = [f'generator://{name}/{random.randint(0, 2**32 - 1)}' for _ in range(train + val + test)]
            append(sampled_programs[:train], train_file, 'Online')
            append(sampled_programs[:min(train, nums_per_benchmark)], it_train_file, 'Online')
            reset(sampled_programs[:min(train, nums_per_benchmark)], f'Benchmark/__train_{name}.txt', 'Online')
            tmp = name.split('-')[0]
            train_list.append(f'{tmp} __train_{name}')
            append(sampled_programs[train: train + val], val_file, 'Online')
            if test > 0:
                test_reset(sampled_programs[train + val: train + val + test], f'Benchmark/final_test_{name}.txt', 'Online')

        for name, train, val, test in local_benchmarks:
            all_programs = get_local_all(name)
            sampled_programs = random.sample(all_programs, min(len(all_programs), train + val + test))
            append(sampled_programs[:train], train_file, 'Local')
            append(sampled_programs[:min(train, nums_per_benchmark)], it_train_file, 'Local')
            reset(sampled_programs[:min(train, nums_per_benchmark)], f'Benchmark/__train_{name}.txt', 'Local')
            append(sampled_programs[train: train + val], val_file, 'Local')
            if test > 0:
                test_reset(sampled_programs[train + val: train + val + test], f'Benchmark/final_test_{name}.txt', 'Local')

        with open('train_list.txt', 'w') as f:
            for line in train_list:
                f.write(f'{line}\n')

        data = []
        with open(train_file, 'r') as f:
            for line in f:
                data.append(line)
        random.shuffle(data)
        with open(search_train_file, 'w') as f:
            for line in data[:17500]:
                f.write(line)


def gen_pss(x, sql_list):
    return ProSeqSiz(x, None, [])


def load_train_pss(name: str, max_p) -> list[ProSeqSiz]:
    filepath = f'{get_root_path()}/Train/Benchmark/{name}.txt'
    data = []

    with open(filepath, 'r') as f:
        for line in f:
            data.append(line.strip())
            max_p -= 1
            if max_p == 0:
                break

    # args = [(line,) for line in data]
    # pss = start_tasks(gen_pss, args, desc='LoadingFiles')
    pss = []
    with ThreadPoolExecutor(max_workers=EnvManager.max_concurrency) as executor:
        futures = [executor.submit(lambda x: ProSeqSiz(x, None, []), line) for line in data]
        with tqdm(total=len(data), desc='Loading local file') as bar:
            for future in tqdm(as_completed(futures)):
                bar.update(1)
                pss.append(future.result())
    return pss


def get_local_all(name):
    res = []
    cf_random = f'{get_root_path()}/DataSet/{name}'
    for root, dirs, files in os.walk(cf_random):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == '.cpp':
                res.append(file)
    return res


def manual(name='codeforces'):
    cf_test = f'Benchmark/{name}_test.txt'
    with open(cf_test, 'w') as _:
        pass

    with open(cf_test, 'a') as f:
        for file in get_local_all(name):
            f.write(f'Local {file}\n')


if __name__ == "__main__":
    random.seed(998244353)
    gen_all()
    pass
