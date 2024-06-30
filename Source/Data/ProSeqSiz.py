import sys
sys.path.append('./../../')

import math
from compiler_gym.datasets import Benchmark

from Source.Env.EnvManager import EnvManager


class ProSeqSiz:
    """
    This class is used to store program and its corresponding sequence.
    """

    INF = float("inf")
    """
    Use inf to replace the code size when program have not been optimized
    """

    def __init__(self, name: str, benchmark, seq: list[int], size=INF):
        """
        :param name: This is the name of the program, formatted as '{Online/Local} name', where Online or Local
        respectively denote programs built into the compiler-gym and those manually added by the users. See
        Source/Train/Benchmark/xxx.train for more details.
        > If type of program is built-in to compiler-gym, just use name as benchmark
        :param benchmark: Element built into compiler-gym
        :param seq: The pass sequence currently used for this program, which is a list of integers
        :param size: Code size of the program
        """
        self.__name: str = name
        if benchmark is None:
            self.__benchmark = EnvManager.load_benchmark(name)
        else:
            self.__benchmark = benchmark
        self.__sequence: list[int] = seq
        self.Init = ProSeqSiz.INF
        self.O0 = ProSeqSiz.INF
        self.Oz = ProSeqSiz.INF
        self.Size = size

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name: str):
        self.Size = ProSeqSiz.INF
        self.__name = name

    @property
    def benchmark(self):
        return self.__benchmark

    @benchmark.setter
    def benchmark(self, benchmark: Benchmark):
        self.Size = ProSeqSiz.INF
        self.__benchmark = benchmark

    @property
    def sequence(self):
        return self.__sequence

    @sequence.setter
    def sequence(self, seq: list[int]):
        self.Size = ProSeqSiz.INF
        self.__sequence = seq

    def __lt__(self, other):
        """
        compare two ProSeqSiz objects
        First Key: code size
        Second Key: length of sequence
        """
        # the smaller, the better
        if not math.isclose(self.get_size(), other.get_size()):
            return self.get_size() < other.get_size()
        return len(self.sequence) < len(other.sequence)

    def __le__(self, other):
        """
        Similar to __lt__
        :param other:
        :return:
        """
        if not math.isclose(self.get_size(), other.get_size()):
            return self.get_size() < other.get_size()
        return len(self.sequence) <= len(other.sequence)

    def __eq__(self, other):
        """
        Similar to __lt__, but sequences is required to completely consistent
        :param other:
        :return:
        """
        return (math.isclose(self.get_size(), other.get_size())
                and self.name == other.name
                and self.sequence == other.sequence)

    def extend(self, seq: list[int]):
        """
        Extend seq after self.sequence
        :param seq:
        :return:
        """
        self.Size = ProSeqSiz.INF
        self.sequence.extend(seq)

    def get_init(self):
        if self.Init == ProSeqSiz.INF:
            self.Init = EnvManager.evaluate_init(self.export())
        return self.Init

    def get_O0(self) -> float:
        """
        Get code size of optimizing this program with "-O0"
        :return: code size
        """
        pass

    def get_Oz(self) -> float:
        """
        Get code size of optimizing program with "-Oz"
        :return: code size
        """
        if self.Oz == ProSeqSiz.INF:
            self.Oz = EnvManager.evaluate_Oz(self.export())
        return self.Oz

    def get_size(self) -> float:
        """
        Get current code size of program
        :return: code size
        """
        if self.Size == ProSeqSiz.INF:
            self.Size = EnvManager.evaluate(self.export(), sql_list=None)
        else:
            pass
        return self.Size

    def get_geo(self) -> float:
        """
        Return geometric mean, which is defined as stated in the expression below
        :return:
        """
        return 1.0 if math.isclose(self.get_size(), 0) else self.get_Oz() / self.get_size()

    def get_arith(self) -> float:
        """
        Return arith mean, which is defined as stated in the expression below
        :return:
        """
        return 0.0 if math.isclose(self.get_Oz(), 0) else (self.get_Oz() - self.get_size()) / self.get_Oz()

    def export(self):
        """
        Export class as tuple form for EnvManager processing
        """
        return self.__name, self.__benchmark, self.__sequence

    @staticmethod
    def import_pss(pss_tuple, size=INF):
        """
        Reverse process of export, pass code size to avoid redundant computations
        """
        return ProSeqSiz(pss_tuple[0], pss_tuple[1], pss_tuple[2], size)

