import sys
sys.path.append('./../../')
from pathlib import Path

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Evaluation.model_train_eval import ModelTrainEval
import argparse


class EvalPrediction:
    def __init__(self, train_benchmark: Benchmark, val_benchmark: Benchmark, test_benchmark: Benchmark,
                 coreset: Coreset):
        self.train_benchmark = train_benchmark
        self.val_benchmark = val_benchmark
        self.test_benchmark = test_benchmark
        self.coreset = coreset
        print('size of coreset', len(self.coreset.seqs))

    def run(self, feature_type='Lookahead'):
        model_train_eval = ModelTrainEval(self.train_benchmark,
                                          self.val_benchmark,
                                          self.test_benchmark,
                                          self.coreset)
        model_path = model_train_eval.train(feature_type=feature_type)
        model_train_eval.eval(model_path, feature_type=feature_type)


if __name__ == '__main__':
    for coreset_name in ['bpc_coreset', 'nvp_coreset_1', 'nvp_coreset_2', 'gens_coreset']:
        for feature_type in ['Autophase', 'InstCountNorm']:
            train_eval = EvalPrediction(
                Benchmark('__train', '__train'),
                Benchmark('__val', '__val'),
                Benchmark('__test', '__test'),
                Coreset(name=coreset_name, need_unique=False),
            )
            train_eval.run(feature_type)
