import copy
import os

from sklearn.cluster import SpectralClustering
import numpy as np

from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Util.parallel import start_tasks
from Source.Util.util import get_in_degree_of_graph, str_to_int_list
from Source.Util.util import get_root_path


class SubSeqGen:

    def __init__(self, name: str, O, P, K, c):
        self.name = name
        self.O: list[int] = O
        self.P: list[ProSeqSiz] = P
        self.K: int = K
        self.c: int = c
        self.subseq_set = []

    def get_identify(self):
        return f'O{len(self.O)}P{len(self.P)}K{self.K}c{self.c}'

    # Calculate its harmonic average speedup on the whole program set P
    def solve1(self, passes, sql_list):
        def trans(pss_tmp: ProSeqSiz, seq_tmp: list[int]) -> tuple:
            pro_seq2 = copy.deepcopy(pss_tmp)
            pro_seq2.sequence = seq_tmp
            return pro_seq2.export()

        env = EnvManager()
        size_list = []
        for pss in self.P:
            tmp = trans(pss, passes)
            size = env.evaluate_for_mp(tmp, sql_list)
            size_list.append(size)

        return size_list

    def gen(self):

        def geo(v1: float, v2: float) -> float:
            return 1.0 if v2 == 0 else v1 / v2

        init_size_task = [([], )]
        init_size = start_tasks(self.solve1, init_size_task, desc='Init Size')[0]
        single_pass_size_task = [([o], ) for o in self.O]
        single_pass_size = start_tasks(self.solve1, single_pass_size_task, desc='Single Pass Size')

        f = []
        for init, ls in zip(init_size, single_pass_size):
            f.append(len(self.P) / sum([geo(v, init) for v in ls]))

        # Keep the top K passes O∗ with the sorted average speedup f from high to low
        self.K = min(self.K, len(f))
        O_star = [ele for _, ele in sorted(zip(f, self.O), reverse=True)][:self.K]

        # Initialize all elements of the collaboration matrix M with 0
        # And build Matrix M
        M = [[0 for _ in range(self.K)] for _ in range(self.K)]

        double_pass_size = []
        for j in range(self.K):
            double_pass_size_task = [([O_star[j], O_star[k]], ) for k in range(self.K)]
            double_pass_size.append(start_tasks(self.solve1, double_pass_size_task, desc=f'Double {j+1}/{self.K}'))

        for idx, _ in enumerate(self.P):
            f_init = init_size[idx]
            f_single = [single_pass_size[e][idx] for e in O_star]
            for j in range(self.K):
                for k in range(self.K):
                    if j == k:
                        continue
                    f_jk = double_pass_size[j][k][idx]
                    f_k = f_single[k]
                    f_j = f_single[j]
                    f_kj = double_pass_size[k][j][idx]
                    if f_jk < f_init and f_jk < min(f_k, f_j, f_kj):
                        M[j][k] += 1

        # Construct a dependency graph G according to the collaboration matrix M
        G = copy.deepcopy(M)
        for i in range(self.K):
            for j in range(i, self.K):
                G[i][j] = G[j][i] = G[i][j] + G[j][i]

        # Segment the dependency graph G into c subgraphs using a graph cut algorithm
        clustering = SpectralClustering(
            n_clusters=self.c,
            affinity='precomputed',
            assign_labels='discretize',
            random_state=0,
        ).fit(np.array(G))
        labels = clustering.labels_
        print(labels)

        # Stretch each subgraph as a pass subsequence with a depth-first traversal algorithm
        ind = get_in_degree_of_graph(M)
        for i in range(self.c):
            subgraph = [idx for idx in range(self.K) if labels[idx] == i]
            root = -1
            for candidate in subgraph:
                if root == -1 or ind[candidate] < ind[root]:
                    root = candidate

            # clear visit array
            visit = [False for _ in range(self.K)]
            subseq = []

            def dfs(u: int):
                subseq.append(O_star[u])
                visit[u] = True
                next_v = -1
                for v in range(self.K):
                    if labels[u] == labels[v] and not visit[v]:
                        if next_v == -1 or M[u][v] > M[u][next_v]:
                            next_v = v
                if next_v != -1:
                    dfs(next_v)

            dfs(root)
            self.subseq_set.append(subseq)

        print(f'subseq_set: {self.subseq_set}')

    def save(self):
        prefix = f'{get_root_path()}/OtherWork/ICMC'
        filepath = f'{prefix}/{self.name}_{self.get_identify()}.txt'
        with open(filepath, 'w') as f:
            for subseq in self.subseq_set:
                f.write(f'{subseq}\n')

    def load(self):
        prefix = f'{get_root_path()}/OtherWork/ICMC'
        filepath = f'{prefix}/{self.name}_{self.get_identify()}.txt'
        if not os.path.exists(filepath):
            print(f'Load from file failed.')
            return False
        with open(filepath, 'r') as f:
            for line in f:
                subseq = str_to_int_list(line)
                if subseq:
                    self.subseq_set.append(subseq)
        print(f'Load from file succeeded.')
        return True
