import copy
from dataclasses import dataclass

from Source.DataBase.DBWrapper import DB
from Source.Env.EnvManager import EnvManager
from Source.Util.util import str_to_int_list, get_root_path, my_mkdir


@dataclass
class BoostPassSet:
    """
    This class stores a 'Boost Structure' collection.
    Examples:
    [
        [1, 2, 3],
        [4, 5, 6],
        [2, 3, 111],
    ]
    """
    seq_set: list[list[int]]

    def __init__(self, elems: list[list[int]]):
        self.seq_set = elems
        self.unique()

    def __iter__(self):
        yield from self.seq_set

    def append(self, e: list[int]):
        """
        Append a Boost Structure into collection
        """
        self.seq_set.append(e)

    def extend(self, ls: list[list[int]]):
        """
        Extend some Boost Structure into collection
        """
        self.seq_set.extend(ls)

    def unique(self):
        """
        Deduplication.
        """
        self.seq_set = [list(e) for e in set(tuple(e) for e in self.seq_set)]

    def __add__(self, other):
        """
        The same to extend.
        """
        self.seq_set.extend(other.seq_set)
        self.unique()
        return copy.deepcopy(self)

    def __eq__(self, other):
        tmp1 = copy.deepcopy(self.seq_set)
        tmp2 = copy.deepcopy(other.seq_set)
        tmp1.sort()
        tmp2.sort_by_name()
        return tmp1 == tmp2

    def load_from_db(self, table_name):
        if not DB.db.exist_table(table_name):
            return False
        res = DB.db.select_all(table_name)
        if res is None:
            return False
        else:
            # print(f'res: {res}')
            self.seq_set = []
            for seq in res:
                self.seq_set.append(str_to_int_list(seq[0]))
            return True

    def save_to_db(self, table_name):
        DB.db.delete_table_if_exist(table_name)
        DB.db.create_table(table_name,
                                   ['seq'],
                                   ['TEXT'],
                                   ['seq'])

        for seq in self.seq_set:
            DB.db.insert(table_name, [seq])
        return True

    def save(self, algo_name: str, iteration: int, name):
        dir_path = f'{get_root_path()}/Result/{algo_name}/{name}/'
        my_mkdir(dir_path)
        filepath = f'{dir_path}/iter_{iteration}.actions'
        with open(filepath, 'w') as f:
            for seq in self.seq_set:
                # print(pss)
                f.write(f'{seq}\n')


@dataclass
class Test:
    ls = list[int]

    def __init__(self):
        self.ls = [1, 2, 3]

    def __iter__(self):
        yield from self.ls


if __name__ == '__main__':
    # bs = BoostPassSet([[1], [1], [1], [1, 2]])
    # print(bs)
    test = Test()
    for i in test:
        print(i)
    pass
