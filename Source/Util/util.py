import glob
import os
import statistics
from pathlib import Path

import math
import numpy as np


class GLOBAL_VALUE:
    INIT = -2
    OZ = -1
    MULTIPROCESSING = 0
    RAY = 1
    PARALLEL_STRATEGY = MULTIPROCESSING
    INF = int(2e9)
    # PARALLEL_STRATEGY = RAY
    TABLE_ORDER = [
        'anghabench-v1',
        'blas-v0',
        'github-v0',
        'linux-v0',
        'opencv-v0',
        'poj104-v1',
        'tensorflow-v0',
        'clgen-v0',
        'csmith-v0',
        'llvm-stress-v0',
        'cbench-v1',
        'chstone-v0',
        'mibench-v1',
        'npb-v0',
        'codeforces',
        'atcoder',
    ]


def db_str_filter(e):
    return str(f"'{e}'") if isinstance(e, str) else str(e)


def db_filter(e):
    if isinstance(e, str):
        return db_str_filter(e)
    elif isinstance(e, list):
        return db_str_filter(db_list_to_string(e))
    else:
        return e


def db_list_to_string(ls: list, ch=',') -> str:
    ls = [db_str_filter(e) for e in ls]
    return ch.join(ls)


def int_list_to_string(int_list: list[int], ch=',') -> str:
    return ch.join(list(map(str, int_list)))


def str_list_to_string(str_list: list[str], ch=',') -> str:
    return ch.join(list(map(str, str_list)))


def get_root_path() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_project_path() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def unique(ls: list) -> list:
    return [list(e) for e in set(tuple(e) for e in ls)]


def str_to_int_list(s: str) -> list:
    s = (s.replace('[', '')
         .replace(']', '')
         .replace('(', '')
         .replace(')', '')
         .replace('"', '')
         .replace("'", '')
         .replace(' ', '')
         .replace('\n', '')
         .replace('\t', ''))
    if s == '':
        return []
    else:
        return list(map(int, s.split(',')))


def str_to_float_list(s: str) -> list:
    s = (s.replace('[', '')
         .replace(']', '')
         .replace('(', '')
         .replace(')', '')
         .replace('"', '')
         .replace("'", '')
         .replace(' ', '')
         .replace('\n', '')
         .replace('\t', ''))
    if s == '':
        return []
    else:
        return list(map(float, s.split(',')))


def my_mkdir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def get_in_degree_of_graph(M):
    ar = np.array(M)
    ind = [ar[:, i].sum() for i in range(ar.shape[1])]
    return ind

def find_file(directory, filename):
    # 构造文件路径的模式匹配字符串
    pattern = os.path.join(directory, '**', filename)
    # 使用glob.glob()函数获取匹配的文件列表
    matches = glob.glob(pattern, recursive=True)
    # 如果有匹配的文件，则返回第一个匹配的文件路径
    if matches:
        return matches[0]
    # 如果未找到匹配的文件，返回None
    return None


def load_local_cpp(name):
    path = f'{get_root_path()}/DataSet'
    return find_file(path, filename=name)


def eval_not_empty_line(file_path):
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def read_coreset_to_file(filepath) -> list[list[int]]:
    coreset = []
    if type(filepath) is Path:
        with filepath.open(mode='r') as f:
            for line in f:
                coreset.append(str_to_int_list(line))
    else:
        with open(filepath, mode='r') as f:
            for line in f:
                coreset.append(str_to_int_list(line))
    return coreset


def save_coreset_to_file(coreset: list[list[int]], filepath):
    if type(filepath) is Path:
        with filepath.open(mode='w') as f:
            for seq in coreset:
                f.write(f'{seq}\n')
    else:
        with open(filepath, mode='w') as f:
            for seq in coreset:
                f.write(f'{seq}\n')


def load_train_list(filepath):
    res = []
    with open(filepath, 'r') as f:
        for line in f:
            name, path = line.strip().split(' ')
            res.append((name, path))
    return res


def get_geo(size, oz) -> float:
    """
    Return geometric mean, which is defined as stated in the expression below
    :return:
    """
    return 1.0 if math.isclose(size, 0) else oz / size


def get_arith(size, oz) -> float:
    """
    Return arith mean, which is defined as stated in the expression below
    :return:
    """
    return 0.0 if math.isclose(oz, 0) else (oz - size) / oz


def get_arith_mean(values):
    """
    Get the geometric mean of optimisation value
    :return:
    """
    return statistics.mean(values)


def get_geo_mean(values):
    """
    Get the geometric mean of optimisation value
    :return:
    """
    return statistics.geometric_mean(values)


def get_benchmark_name(name):
    if 'benchmark:' in name or 'generator:' in name:
        return name.split('/')[2]
    elif 'abc' in name and '.cpp' in name:
        return 'atcoder'
    elif '.cpp' in name:
        return 'codeforces'
