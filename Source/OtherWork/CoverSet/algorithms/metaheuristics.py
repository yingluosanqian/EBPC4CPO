"""
Copyright 2021 Anderson Faustino da Silva

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from dataclasses import dataclass
import random

import pygmo as pg

from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans
from Source.OtherWork.CoverSet.essentials.sequence import Sequence

TOTAL_COUNT = 0

class Pygmo:
    """A Pygmo's strategy."""

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #       'seq': list}}
    __results = None

    # SGA
    # {gen = {'fevals': int,
    #         'best': float,
    #         'improvement': float}}
    #
    # PSO
    # {gen: {'fevals': int,
    #        'gbest': float,
    #        'meanvel': float,
    #        'meanlbest': float,
    #        'avgdist': float}
    __log = None

    class Problem:
        """Pygmo's problem."""

        def __init__(self,
                     dimension,
                     env: EnvManager,  # compiler -> env
                     pss: ProSeqSiz, ):
            """Construct a Pygmo problem.

            Arguments
            ----------

            dimension : int
                The length of a sequence.

            env : EnvManager()

            pss : ProSeqSiz
            """
            self.dimension = dimension
            self.env = env
            self.pss = pss
            self.sql_list = []

        def __deepcopy__(self,
                         *args,
                         **kwargs):
            return self

        def fitness(self,
                    sequence):
            """Calculate and return the fitness."""
            # global TOTAL_COUNT
            # TOTAL_COUNT += 1
            # print(TOTAL_COUNT)
            sequence = Sequence.fix_index(list(sequence))
            sequence = Sequence.sanitize(sequence)
            goal_value = self.env.evaluate_for_mp(Trans.copy_and_reset(self.pss, sequence), self.sql_list)
            return [goal_value]

        def get_nix(self):
            """Integer dimension of the problem"""
            return self.dimension

        def get_bounds(self):
            """Box-bounds"""
            return ([0] * self.dimension,
                    [123] * self.dimension)

        def get_name(self):
            """Problem name"""
            return 'Optimization Selection'

        def get_extra_info(self):
            """Info"""
            return '\tDimensions: ' + str(self.dimension)

    @dataclass
    class PygmoFlags:
        """Pygmo flags

        Arguments
        ----------

        dimension : int
            The length of a sequence.

        population : int

        env : EnvManager

        pss : ProSeqSiz
        """
        dimension: int
        population: int
        env: EnvManager
        pss: ProSeqSiz

    def __init__(self,
                 dimension,
                 population,
                 env,
                 pss):
        """Initialize the arguments.

        Arguments
        ----------
        dimension : int
            The length of a sequence.

        population : int

        env : EnvManager

        pss : ProSeqSiz
        """

        self.__flags = self.PygmoFlags(dimension,
                                       population,
                                       env,
                                       pss)

    @property
    def results(self):
        """Getter"""
        return self.__results

    @property
    def log(self):
        """Getter"""
        return self.__log

    def exec(self, sga):
        """Execute the algorithm.

        Argument
        ---------
        algorithm : Pygmo algorithm

        benchmark : str
        """
        # Step 1: Algorithm
        algorithm = pg.algorithm(sga)
        # algorithm.set_verbosity(1)

        # Step 2: Instantiate a pygmo problem
        my_problem = self.Problem(self.__flags.dimension,
                                  self.__flags.env,
                                  self.__flags.pss)
        problem = pg.problem(my_problem)

        # Step 3: The initial population
        population = pg.population(problem,
                                   self.__flags.population)

        # Step 4: Evolve the population
        population = algorithm.evolve(population)

        # Step 5: Get the results
        sga_sequence = population.get_x().tolist()
        sga_fitness = population.get_f().tolist()

        pss_list = []
        for index in range(self.__flags.population):
            sequence = sga_sequence[index]
            sequence = Sequence.fix_index(list(sequence))
            goal_value = sga_fitness[index][0]
            if goal_value == float('inf'):
                continue
            # self.__results[index] = {'seq': sequence,
            #                          'goal': goal_value}

            pss_list.append(ProSeqSiz.import_pss(Trans.copy_and_reset(self.__flags.pss, sequence)))

        return pss_list, my_problem.sql_list
        # # Step 6: Get the log
        # self.__log = {}
        # if algorithm.get_name() == 'SGA: Genetic Algorithm':
        #     uda = algorithm.extract(pg.sga)
        #     log = uda.get_log()
        #     for (gen, fevals, best, improvement) in log:
        #         self.__log[gen] = {'fevals': fevals,
        #                            'best': best,
        #                            'improvement': improvement}
        # elif algorithm.get_name() == 'PSO: Particle Swarm Optimization':
        #     uda = algorithm.extract(pg.pso)
        #     log = uda.get_log()
        #     for (gen, fevals, gbest, meanvel, meanlbest, avgdist) in log:
        #         self.__log[gen] = {'fevals': fevals,
        #                            'gbest': gbest,
        #                            'meanvel': meanvel,
        #                            'meanlbest': meanlbest,
        #                            'avgdist': avgdist}


class SGA(Pygmo):
    """Simple Genetic Algorithm"""

    __version__ = '1.0.0'

    __flags = None

    @dataclass
    class Flags:
        """Pygmo flags

        Arguments
        ----------
        generations : int

        cr : float
            Crossover probability

        m : float
            Mutation probability

        param_m  : float
            Distribution index (polynomial mutation),
            gaussian width (gaussian mutation) or
            inactive (uniform mutation)

        param_s : float
            The number of best individuals to use in “truncated”
            selection or the size of the tournament in
            tournament selection.

        crossover : str
            exponential, binomial or single

        mutation : str
            gaussian, polynomial or uniform

        selection : str
            tournament or truncated

        seed : int
        """
        generations: int
        cr: float
        m: float
        param_m: float
        param_s: float
        crossover: str
        mutation: str
        selection: str
        seed: int

    def __init__(self,
                 generations,
                 population,
                 cr,
                 m,
                 param_m,
                 param_s,
                 crossover,
                 mutation,
                 selection,
                 seed,
                 dimension,
                 env,
                 pss):
        """Constructor

        Arguments
        ----------
        generations : int

        population : int

        cr : float
            Crossover probability

        m : float
            Mutation probability

        param_m  : float
            Distribution index (polynomial mutation),
            gaussian width (gaussian mutation) or
            inactive (uniform mutation)

        param_s : float
            The number of best individuals to use in “truncated”
            selection or the size of the tournament in
            tournament selection.

        crossover : str
            exponential, binomial or single

        mutation : str
            gaussian, polynomial or uniform

        selection : str
            tournament or truncated

        seed : int

        dimension : int
            The length of a sequence.

        env : Manager

        pss: ProSeqSiz
        """
        self.__flags = self.Flags(generations,
                                  cr,
                                  m,
                                  param_m,
                                  param_s,
                                  crossover,
                                  mutation,
                                  selection,
                                  seed)
        super().__init__(dimension,
                         population,
                         env,
                         pss)

    def run(self):
        """Execute the algorithm

        Argument
        --------
        """
        sga = pg.sga(gen=self.__flags.generations,
                     cr=self.__flags.cr,
                     m=self.__flags.m,
                     param_m=self.__flags.param_m,
                     param_s=self.__flags.param_s,
                     crossover=self.__flags.crossover,
                     mutation=self.__flags.mutation,
                     selection=self.__flags.selection,
                     seed=self.__flags.seed)

        # Execute
        return super().exec(sga)


class PSO(Pygmo):
    """Particle Swarm Optimization."""

    __version__ = '1.0.0'

    __flags = None

    @dataclass
    class Flags:
        """PSO flags

        Arguments
        ----------
        generations : int

        omega : float
            Inertia weight (or constriction factor)

        eta1 : float
            Social component

        eta2 : float
            Cognitive component

        max_vel : float
            Maximum allowed particle velocities
            (normalized with respect to the bounds width)

        variant : int
            Algorithmic variant

        neighb_type : int
            Swarm topology (defining each particle’s neighbours)

        neighb_param : int
            Topology parameter (defines how many neighbours to consider)

        memory : bool
            When true the velocities are not reset between successive
            calls to the evolve method

        seed : int
            Seed used by the internal random number generator.
        """
        generations: int
        omega: float
        eta1: float
        eta2: float
        max_vel: float
        variant: int
        neighb_type: int
        neighb_param: int
        memory: bool
        seed: int

    def __init__(self,
                 generations,
                 population,
                 omega,
                 eta1,
                 eta2,
                 max_vel,
                 variant,
                 neighb_type,
                 neighb_param,
                 memory,
                 seed,
                 dimension,
                 passes_filename,
                 goals,
                 compiler,
                 benchmarks_directory,
                 working_set,
                 times,
                 tool,
                 verify_output):
        """Constructor

        Arguments
        ----------
        generations : int

        population : int

        omega : float
            Inertia weight (or constriction factor)

        eta1 : float
            Social component

        eta2 : float
            Cognitive component

        max_vel : float
            Maximum allowed particle velocities
            (normalized with respect to the bounds width)

        variant : int
            Algorithmic variant

        neighb_type : int
            Swarm topology (defining each particle’s neighbours)

        neighb_param : int
            Topology parameter (defines how many neighbours to consider)

        memory : bool
            When true the velocities are not reset between successive
            calls to the evolve method

        seed : int
            Seed used by the internal random number generator.
        """
        self.__flags = self.Flags(generations,
                                  omega,
                                  eta1,
                                  eta2,
                                  max_vel,
                                  variant,
                                  neighb_type,
                                  neighb_param,
                                  memory,
                                  seed)

        super().__init__(dimension,
                         population,
                         passes_filename,
                         goals,
                         compiler,
                         benchmarks_directory,
                         working_set,
                         times,
                         tool,
                         verify_output)

    def run(self):
        """Execute the algorithm

        Argument
        --------
        benchmark : str
        """
        if PSO.__flags.seed:
            algorithm = pg.pso(self.__flags.generations,
                               self.__flags.omega,
                               self.__flags.eta1,
                               self.__flags.eta2,
                               self.__flags.max_vel,
                               self.__flags.variant,
                               self.__flags.neighb_type,
                               self.__flags.neighb_param,
                               self.__flags.memory,
                               self.__flags.seed)
        else:
            algorithm = pg.pso(self.__flags.generations,
                               self.__flags.omega,
                               self.__flags.eta1,
                               self.__flags.eta2,
                               self.__flags.max_vel,
                               self.__flags.variant,
                               self.__flags.neighb_type,
                               self.__flags.neighb_param,
                               self.__flags.memory)

        # Execute
        super().exec(algorithm)
