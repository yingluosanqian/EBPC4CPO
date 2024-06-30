import sys
sys.path.append('./../../')
import copy
from pathlib import Path

import numpy as np
import torch

from Source.Data.Benchmark import Benchmark
from Source.Data.Coreset import Coreset
from Source.Data.ProSeqSiz import ProSeqSiz
from Source.Env.EnvManager import EnvManager
from Source.Evaluation.MLP.MLP4Autophase import load_train, load_val, MLP, mlp_training, mlp_testing, model_saving, \
    model_loading
from Source.Evaluation.evaluation import Trans
from Source.Evaluation.model_evaluation import ModelEvaluation
from Source.Util.parallel import start_tasks
from Source.Util.util import get_root_path


class ModelTrainEval:

    def __init__(self, train_benchmark: Benchmark, val_benchmark: Benchmark, test_benchmark: Benchmark, coreset: Coreset):
        self.train_benchmark = train_benchmark
        self.val_benchmark = val_benchmark
        self.test_benchmark = test_benchmark
        self.coreset = coreset
        self.data_path = Path('.') / 'MLP' / 'data'
        self.data_path.mkdir(parents=True, exist_ok=True)
        pass

    def gen_labels(self, pss: ProSeqSiz, all_seq: list[list[int]], sql_list):
        env = EnvManager()
        init = env.evaluate_for_mp(Trans.copy_and_reset(pss, []), sql_list)
        size_list = []
        for seq in all_seq:
            tmp_pss = Trans.copy_and_reset(pss, seq)
            _, size = env.evaluate_for_mp_with_cut(tmp_pss, sql_list)
            size_list.append(init - size)

        return size_list

    def gen_AutophaseNorm_features_and_labels(self, benchmark: Benchmark):
        features_file_path: Path = self.data_path / f'{self.coreset.name}_{benchmark.name}_autophase_features.csv'
        labels_file_path: Path = self.data_path / f'{self.coreset.name}_{benchmark.name}_autophase_labels.csv'
        available_index = np.array(list(range(len(benchmark.pss_list))))
        if not features_file_path.is_file() or not labels_file_path.is_file():
            # feature
            args = [(pss.benchmark, 'Autophase') for pss in benchmark.pss_list]
            feature_vectors = np.array(start_tasks(EnvManager.get_feature, args, desc='Get Program Features'))
            feature_vectors = feature_vectors / feature_vectors[:, 51].reshape(feature_vectors.shape[0], 1)  # norm
            # label
            args = [(pss, self.coreset.seqs) for pss in benchmark.pss_list]
            label_vectors = np.array(start_tasks(self.gen_labels, args, desc='Get Program Labels'))
            # handle exception
            available_index = np.intersect1d(available_index,
                                             [i for i, b in enumerate(~np.all(feature_vectors == -1, axis=1)) if b])
            print(available_index)
            feature_vectors, label_vectors = feature_vectors[available_index], label_vectors[available_index]
            # save
            np.savetxt(features_file_path, feature_vectors, delimiter=",")
            np.savetxt(labels_file_path, label_vectors, delimiter=",")

        return features_file_path, labels_file_path

    def gen_InstCountNorm_features_and_labels(self, benchmark: Benchmark):
        features_file_path: Path = self.data_path / f'{self.coreset.name}_{benchmark.name}_instcountnorm_features.csv'
        labels_file_path: Path = self.data_path / f'{self.coreset.name}_{benchmark.name}_instcountnorm_labels.csv'
        available_index = np.array(list(range(len(benchmark.pss_list))))
        if not features_file_path.is_file() or not labels_file_path.is_file():
            # feature
            args = [(pss.benchmark, 'InstCountNorm') for pss in benchmark.pss_list]
            feature_vectors = np.array(start_tasks(EnvManager.get_feature, args, desc='Get Program Features'))
            # label
            args = [(pss, self.coreset.seqs) for pss in benchmark.pss_list]
            label_vectors = np.array(start_tasks(self.gen_labels, args, desc='Get Program Labels'))
            # handle exception
            available_index = np.intersect1d(available_index,
                                             [i for i, b in enumerate(~np.all(feature_vectors == -1, axis=1)) if b])
            print(available_index)
            feature_vectors, label_vectors = feature_vectors[available_index], label_vectors[available_index]
            # save
            np.savetxt(features_file_path, feature_vectors, delimiter=",")
            np.savetxt(labels_file_path, label_vectors, delimiter=",")

        return features_file_path, labels_file_path

    def train(self, feature_type):
        if feature_type == 'Autophase':
            fl_gen = self.gen_AutophaseNorm_features_and_labels
        elif feature_type == 'InstCountNorm':
            fl_gen = self.gen_InstCountNorm_features_and_labels
        elif feature_type == 'Lookahead':
            fl_gen = self.gen_lookahead_features_and_labels
        else:
            raise NotImplementedError()

        model_path = Path('MLP') / Path('saved_models')
        model_path.mkdir(parents=True, exist_ok=True)
        model_path = model_path / f'{self.coreset.name}_{feature_type}_mlp'

        if not model_path.is_file():
            np.random.seed(998244353)
            torch.manual_seed(998244353)

            X_train, Y_train = load_train(*fl_gen(self.train_benchmark), T=0.0015)
            X_val, Y_val = load_val(*fl_gen(self.val_benchmark))
            model = MLP(input_size=X_train.shape[1], output_size=Y_train.shape[1])
            mlp_training(model, X_train, Y_train, batch_size=500, num_epochs=100, lr=8e-3)
            mlp_testing(model, X_val, Y_val)
            model_saving(model, model_path)
        return model_path

    def eval(self, model_path, feature_type, output_size=50):
        if feature_type == 'Autophase':
            fl_gen = self.gen_AutophaseNorm_features_and_labels
        elif feature_type == 'InstCountNorm':
            fl_gen = self.gen_InstCountNorm_features_and_labels
        elif feature_type == 'Lookahead':
            raise NotImplementedError()
            # fl_gen = self.gen_lookahead_features_and_labels
        else:
            raise NotImplementedError()

        features_file_path, _ = fl_gen(self.test_benchmark)
        X_test = torch.Tensor(np.loadtxt(features_file_path, delimiter=","))
        model = MLP(input_size=X_test.shape[1], output_size=output_size)
        model_loading(model, model_path)

        ModelEvaluation.evaluate_coreset(
            total_benchmark=copy.deepcopy(self.test_benchmark),
            X_test=X_test,
            coreset=self.coreset,
            model=model,
            model_path=model_path,
            top_list=tuple(range(1, output_size + 1))
        )

    def eval_by_pass(self, model_path, feature_type, output_size=50):
        if feature_type == 'Autophase':
            fl_gen = self.gen_AutophaseNorm_features_and_labels
        elif feature_type == 'InstCountNorm':
            fl_gen = self.gen_InstCountNorm_features_and_labels
        elif feature_type == 'Lookahead':
            fl_gen = self.gen_lookahead_features_and_labels
        else:
            raise NotImplementedError()

        features_file_path, _ = fl_gen(self.test_benchmark)
        X_test = torch.Tensor(np.loadtxt(features_file_path, delimiter=","))
        model = MLP(input_size=X_test.shape[1], output_size=output_size)
        model_loading(model, model_path)

        ModelEvaluation.evaluate_coreset_by_pass(
            total_benchmark=copy.deepcopy(self.test_benchmark),
            X_test=X_test,
            coreset=self.coreset,
            model=model,
            model_path=model_path,
            top_list=tuple(range(1, output_size + 1))
        )
