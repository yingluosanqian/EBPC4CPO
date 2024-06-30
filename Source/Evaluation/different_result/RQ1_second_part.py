import sys
sys.path.append('./../../../')

from Source.Util.util import get_root_path


def load_from_txt(filename):
    first = True
    ls = []
    with open(f'{get_root_path()}/Evaluation/coreset/eval/results_{filename}_detail.txt', 'r') as f:
        for line in f:
            if first:
                first = False
            else:
                line = line.split('~')[3]
                v = float(line.split(':')[1])
                ls.append(v)
    return ls


def get_better_num(ls1, ls2):
    count = 0
    for v1, v2 in zip(ls1, ls2):
        # print(v1, v2)
        if v1 > v2:
            count += 1
    return count


def eval(filename_list):
    results = [load_from_txt(filename) for filename in filename_list]
    total = len(results[0])
    results.append([1] * total)
    # print('results:', results)
    labels = ['BPC', 'NVP-1', 'NVP-2', 'ICMC', 'GA', 'LLVM -Oz']
    for ls1, label in zip(results, labels):
        print(f'\\textbf' + '{' + label + '}', end='')
        for ls2 in results:
            print(' & ', end='')
            print(f'{(get_better_num(ls1, ls2) / total) * 100: .1f}\\%', end='')
        print(' \\\\')


if __name__ == '__main__':
    eval([
        f'bpc_coreset.txt',
        f'nvp_coreset_1.txt',
        f'nvp_coreset_2.txt',
        f'icmc_coreset.txt',
        f'ga_coreset.txt',
    ])
