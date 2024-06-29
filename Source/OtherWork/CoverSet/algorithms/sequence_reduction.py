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
import copy
import os

from dataclasses import dataclass

from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.evaluation import Trans
from Source.OtherWork.CoverSet.essentials.sequence import Sequence


class SequenceReduction:
    """Create a small sequence."""

    __version__ = '1.0.0'

    __flags = None

    # {key: {'goal': float,
    #        'seq': list}}
    # {
    #   0: {'seq': list, 'goal': float} # original sequence
    #   1: {'seq': list, 'goal': float} # small sequence
    # }
    __results = {}

    @dataclass
    class Flags:
        """ReduceSequence flags
        env: EnvManager
            The compiler to use.

        pss: ProSeqSiz
        """
        env: EnvManager
        pss: ProSeqSiz

    def __init__(self,
                 env,
                 pss):
        """Constructor

        Arguments
        ---------
        env: EnvManager
            The compiler to use.

        pss: ProSeqSiz
        """
        self.__flags = self.Flags(env, pss)
        self.sql_list = []

    @property
    def results(self):
        """Getter"""
        return self.__results

    def run(self):
        """Sequence Reduction algorithm.

        Suresh Purini and Lakshya Jain.
        Finding Good Optimization Sequences Covering Program Space.
        TACO.
        2013

        Argument
        --------
        sequence : list
        """
        # Calculate the initial value of the goal.
        goal_value = self.__flags.env.evaluate_for_mp(self.__flags.pss.export(), self.sql_list)

        # Sequence Reduction algorithm
        lst_best_sequence = self.__flags.pss.sequence.copy()
        best_goal_value = goal_value
        change = True
        while change:
            change = False
            bestseqlen = len(lst_best_sequence)
            for i in range(bestseqlen):
                vector = [1 for i in range(bestseqlen)]
                vector[i] = 0
                lst_new_sequence = Sequence.remove_passes(
                    lst_best_sequence,
                    vector
                )
                goal_value = self.__flags.env.evaluate_for_mp(Trans.copy_and_reset(self.__flags.pss, lst_new_sequence), self.sql_list)
                if goal_value <= best_goal_value:
                    best_goal_value = goal_value
                    lst_best_sequence = lst_new_sequence[:]
                    change = True
                    break

        return ProSeqSiz.import_pss(Trans.copy_and_reset(self.__flags.pss, lst_best_sequence))
