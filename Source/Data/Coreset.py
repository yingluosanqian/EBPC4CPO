from Source.Util.util import get_root_path, my_mkdir, str_to_int_list, unique


# Coreset

class Coreset:

    def __init__(self, name, seq_list: list[list[int]] = None, need_unique=True):
        self.seqs = []
        self.name = name
        if seq_list is not None:
            self.seqs = seq_list
        else:
            my_mkdir(f'{get_root_path()}/Evaluation/coreset')
            with open(f'{get_root_path()}/Evaluation/coreset/{name}.txt', 'r') as f:
                for line in f:
                    ls = str_to_int_list(line)
                    if ls:
                        self.seqs.append(ls)
            avg = sum([len(seq) for seq in self.seqs]) / len(self.seqs)
            print(f'average length of sequences: {avg}')
        if need_unique:
            self.seqs = unique(self.seqs)

    def __iter__(self):
        # 返回self.seqs的迭代器
        return iter(self.seqs)


class ExCoreset:

    def __init__(self, name):
        self.seq_and_name = {}
        my_mkdir(f'{get_root_path()}/Evaluation/coreset')
        with open(f'{get_root_path()}/Evaluation/coreset/{name}', 'r') as f:
            for line in f:
                ls, name = line.strip().split('&')
                ls = str_to_int_list(ls)
                if ls:
                    ls = tuple(ls)
                    if ls not in self.seq_and_name.keys():
                        self.seq_and_name[ls] = []
                    self.seq_and_name[ls].append(name)

    def get(self, key, default=None):
        if type(key) is list:
            key = tuple(key)
        return self.seq_and_name.get(key, default)

    def keys(self):
        return [list(seq) for seq in self.seq_and_name.keys()]
