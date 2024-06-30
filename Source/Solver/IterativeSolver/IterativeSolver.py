import sys
sys.path.append('./../../../')
import copy
from Source.Data.Benchmark import Benchmark
from Source.Data.BoostPassSet import BoostPassSet
from Source.Data.Coreset import Coreset
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import EvalCoreset
from Source.Solver.CoresetGen.CoresetGen import CoresetGen
from Source.Solver.Extractor.Extractor import Extractor
from Source.Solver.Greedy.BoostGreedy import BoostGreedy
from Source.Solver.ProbDD.ProbDD import ProbDD
from Source.Solver.Spread.Spread import Spread
from Source.Util.util import get_root_path
import argparse


class IterativeSolver:

    def __init__(self,
                 benchmark: Benchmark,
                 greedy_solver,
                 spread_solver,
                 probdd_solver,
                 extractor_solver,
                 test_benchmark):
        self.benchmark = benchmark
        self.extractor_solver = extractor_solver
        self.spread_solver = spread_solver
        self.greedy_solver = greedy_solver
        self.probdd_solver = probdd_solver
        self.test_benchmark = test_benchmark

    def work(self, iterations: int, algo_name: str):
        boost_passes = [[i] for i in range(EnvManager.get_action_space())]
        last_benchmark = None
        for iteration in range(iterations):
            print(f'Iteration {iteration}:')
            # always put all 124 passes into bpc set
            boost_passes.extend([[i] for i in range(EnvManager.get_action_space())])

            name = f'{self.benchmark.name}_{algo_name}_{iteration}'
            # Iteration.
            bs = BoostPassSet(boost_passes)

            # first step: forward-backward greedy, build pass sequences
            self.greedy_solver.solve(self.benchmark, bs, f'{name}_greedy')

            # second step: remove suboptimal sequence
            if 'no_removal' not in algo_name:
                self.spread_solver.solve(self.benchmark, f'{name}_removal')

            # third step: trim sequence
            if 'no_trimming' not in algo_name:
                self.probdd_solver.solve(self.benchmark, f'{name}_trimming')

            # fourth step: extract BPCs from pass sequences
            boost_passes = self.extractor_solver.solve(self.benchmark, f'{name}_boost')

            # Keep optimal version during iteration.
            print(f'benchmark merge...')
            self.benchmark.merge_min(last_benchmark)
            last_benchmark = copy.deepcopy(self.benchmark)

            # Log.
            print(f'benchmark saving...')
            self.benchmark.save(algo_name, iteration)
            print(f'benchmark export seqs...')
            self.benchmark.export_seqs(algo_name, iteration)
            print(f'boost structure saving...')
            bs.save(algo_name, iteration, self.benchmark.name)  # save bpc set

            # coreset gen
            # Do CoresetGen first, because Refine pass sequence module would reduce the effect of coreset
            temp_benchmark = copy.deepcopy(self.benchmark)
            prefix = f'{get_root_path()}/Result/{algo_name}/{temp_benchmark.name}'
            coreset_gen = CoresetGen(self.benchmark)
            coreset = coreset_gen.gen_coreset(f'{get_root_path()}/Result/{algo_name}/{temp_benchmark.name}/{temp_benchmark.name}_coreset_{iteration}.txt')
            EvalCoreset.evaluate(copy.deepcopy(self.test_benchmark), coreset, f'{prefix}/{self.test_benchmark.name}_result_{iteration}.txt')

        # last round BPC set
        bs.save(algo_name, iterations, self.benchmark.name)  # save bpc set


def parse():
    parser = argparse.ArgumentParser(description="Parse command line arguments")
    parser.add_argument('-n', '--name', required=True, type=str, help='Name of the algorithm (must be one of "base", "no_removal", "no_trimming")')
    parser.add_argument('-i', '--iter', required=True, type=int, help='Number of iterations')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    solver = IterativeSolver(
        Benchmark('__it_train', '__it_train'),
        BoostGreedy(),
        Spread(),
        ProbDD(),
        Extractor(),
        Benchmark('__test', '__test'),
    )
    solver.work(args.iter, args.name)
