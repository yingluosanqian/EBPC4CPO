import sys
sys.path.append('./../../')

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from Source.DataBase.DBWrapper import DB

import compiler_gym

from Source.Util.util import GLOBAL_VALUE, load_local_cpp, str_to_int_list, str_to_float_list


class EnvManager:
    max_concurrency = os.cpu_count()
    # print(f'max_concurrency: {max_concurrency}')

    n = 124

    def __init__(self):
        self.env = compiler_gym.make(
            "llvm-ic-v0",
            reward_space='IrInstructionCount',
        )

    def __del__(self):
        self.env.close()

    def get_whole_list(self, pss):
        try:
            self.env.reset(benchmark=pss[1])
            init = self.env.observation['IrInstructionCount']
            whole_list = [init]
            if pss[2] == [GLOBAL_VALUE.OZ]:
                oz = self.env.observation['IrInstructionCountOz']
                whole_list = [init, oz]
            else:
                for action in pss[2]:
                    self.env.step(action)
                    whole_list.append(self.env.observation['IrInstructionCount'])
            sentence = DB.db.gen_insert_sentence('ProSeqSiz', [pss[0], pss[2], whole_list])
            return whole_list, sentence
        except Exception as e:
            print(f'=> While env.step(), Error {e} occurs.')
            print(f'=> Error file: {pss[0]}')
            whole_list = [GLOBAL_VALUE.INF for _ in range(len(pss[2]) + 1)]
            sentence = DB.db.gen_insert_sentence('ProSeqSiz', [pss[0], pss[2], whole_list])
            return whole_list, sentence
        finally:
            pass

    def toy_get_ir_sha1(self, benchmark, seq):
        try:
            self.env.reset(benchmark=benchmark)
            self.env.multistep(seq)
            return self.env.ir_sha1, self.env.observation['IrInstructionCount']
        except Exception as e:
            print(f'Exception: {e}')
            return 'Error', -1
        finally:
            pass

    def multi_evaluate_whole_for_mp(self, pss_list: list[tuple], sql_list):
        res = [self.evaluate_whole_for_mp(pss, sql_list) for pss in pss_list]
        return res

    def multi_evaluate_for_mp(self, pss_list: list[tuple], sql_list):
        res = self.multi_evaluate_whole_for_mp(pss_list, sql_list)
        return [e[-1] for e in res]

    def evaluate_oz_for_mp(self, pss: tuple, sql_list: list) -> list[int]:
        assert pss[2] == [GLOBAL_VALUE.OZ]
        db_res = DB.db.select('ProSeqSiz',
                              ['size_list'],
                              ['program', 'sequence'],
                              [pss[0], pss[2]])
        if db_res is not None:
            return str_to_int_list(db_res[0])[-1]
        else:
            whole_list, sentence = self.get_whole_list(pss)
            if sentence is not None:
                sql_list.append(sentence)
            return whole_list[-1]

    def evaluate_whole_for_mp(self, pss: tuple, sql_list: list) -> list[int]:
        db_res = DB.db.select('ProSeqSiz',
                              ['size_list'],
                              ['program', 'sequence'],
                              [pss[0], pss[2]])
        if db_res is not None:
            return str_to_int_list(db_res[0])
        else:
            whole_list, sentence = self.get_whole_list(pss)
            if sentence is not None:
                sql_list.append(sentence)
            return whole_list

    def evaluate_for_mp(self, pss: tuple, sql_list) -> int:
        return self.evaluate_whole_for_mp(pss, sql_list)[-1]

    def evaluate_for_mp_with_cut(self, pss: tuple, sql_list) -> (tuple, int):
        whole = self.evaluate_whole_for_mp(pss, sql_list)
        mn = 0
        for idx, val in enumerate(whole):
            if val < whole[mn]:
                mn = idx
        return (pss[0], pss[1], pss[2][:mn+1]), whole[mn]

    @staticmethod
    def make_env(benchmark):
        """
        Create a llvm env with benchmark
        """
        env = compiler_gym.make(
            "llvm-ic-v0",
            reward_space='IrInstructionCount',
        )
        try:
            env.reset(benchmark=benchmark)
        except Exception as e:
            env.reset(benchmark='benchmark://github-v0/10')
            print(f'make_env failed.')
        finally:
            pass
        return env

    @staticmethod
    def get_feature_inst_count(benchmark, sql_list=None):
        env = EnvManager.make_env(benchmark)
        ar = env.observation['InstCount']
        env.close()
        return ar

    @staticmethod
    def get_feature(benchmark, feature_type, sql_list=None):
        db_res = DB.feature_db.select('ProFea',
                                      ['feature_value'],
                                      ['program', 'feature_type'],
                                      [f'{benchmark}', feature_type])
        if db_res is not None:
            return str_to_float_list(db_res[0])
        else:
            try:
                env = EnvManager.make_env(benchmark)
                ft = env.observation[feature_type]
                env.close()
            except Exception as e:
                if feature_type == 'Autophase':
                    ft = [-1 for _ in range(56)]
                elif feature_type == 'InstCountNorm':
                    ft = [-1 for _ in range(69)]
            ft = list(ft)
            sentence = DB.feature_db.gen_insert_sentence('ProFea', [f'{benchmark}', feature_type, ft])
            if sql_list is not None:
                sql_list.append(sentence)
            else:
                DB.feature_db.execute_and_commit(sentence)
        return ft

    @staticmethod
    def get_feature_autophase(benchmark, sql_list=None):
        env = EnvManager.make_env(benchmark)
        ar = env.observation['Autophase']
        env.close()
        return ar

    @staticmethod
    def get_feature_inst_count_norm(benchmark, sql_list=None):
        env = EnvManager.make_env(benchmark)
        ar = env.observation['InstCountNorm']
        env.close()
        return ar

    @staticmethod
    def get_feature_programl(benchmark, sql_list=None):
        env = EnvManager.make_env(benchmark)
        G = env.observation['Programl']
        env.close()
        return G

    @staticmethod
    def load_benchmark(benchmark_name):
        """
        Create a benchmark according to benchmark_name.
        """
        res = benchmark_name.split(' ')
        # print(res)
        # Tag is either 'Online' or 'Local', respectively responding to built-in programs in compiler-gym and
        # user-define programs
        tag, name = res[0], res[1]
        if tag == 'Online':
            return name
        else:
            benchmark = None
            env = compiler_gym.make(
                "llvm-ic-v0",
                reward_space='IrInstructionCount',
            )
            # Use '-O1' rather than '-O0' because '-O0' would disable most optimisations in llvm opt
            copt = ['-O1', '-Xclang -disable-O0-optnone', '-Xclang -disable-llvm-passes', '--std=c++20']
            file_path = load_local_cpp(name)
            # print(file_path)
            # Find program in local files. This is a lazy, failed design, because it doesn't allow duplicate file name
            # Just make do with it for now. Hope someone fixed it.
            if file_path:
                try:
                    benchmark = env.make_benchmark(file_path, copt)
                    # env.write_bitcode(f"/home/cyh/Project/PassOrdering/Source/DataSet/codeforces/contest/name.bc")
                    # print('write xxx to xxx')
                    if benchmark is None:
                        print('Compiler Failed!')
                    # print('Compiler Accepted.')
                except Exception as _:
                    # Attention !!! Attention !!! Attention !!!
                    # Programs that failed in compiling will be deleted.
                    os.remove(file_path)
                    # print('Compiler Error.')
                finally:
                    env.close()
                    return benchmark
            else:
                print(f'Error File [{name}] does not exist.')
                return benchmark

    @staticmethod
    def get_action_space():
        """
        Get total pass built-in llvm opt.
        This number is 124 in llvm 10.0
        :return:
        """
        if EnvManager.n is None:
            env = compiler_gym.make("llvm-ic-v0")
            EnvManager.n = env.action_space.n
            env.close()
        return EnvManager.n

    @staticmethod
    def get_init_size(name):
        return EnvManager.run((name, name, []))

    @staticmethod
    def run(pss: tuple) -> int:
        """
        Optimize a program with its sequence and return the code size after optimisation by the sequence
        :param pss:  0: name, 1: benchmark, 2: sequence
        :return: code size
        """
        env = EnvManager.make_env(pss[1])
        env.multistep(pss[2])
        res = env.observation['IrInstructionCount']
        env.close()
        return res

    @staticmethod
    def run_whole(pss: tuple, env=None, sql_list=None) -> list[int]:
        """
        Optimize a program with its sequence and return the code size after optimisation by the sequence
        :param pss:  0: name, 1: benchmark, 2: sequence
        :param env:
        :return: code size after each step, which formed a list with, whose length equal to len(sequence) + 1
        (e.g. include initial code size)
        """
        if env is None:
            env = EnvManager.make_env(pss[1])
        try:
            env.reset(benchmark=pss[1])
            init = env.observation['IrInstructionCount']
            whole_list = [init]
            if pss[2] == [GLOBAL_VALUE.OZ]:
                oz = env.observation['IrInstructionCountOz']
                whole_list = [init, oz]
            else:
                for action in pss[2]:
                    env.step(action)
                    whole_list.append(env.observation['IrInstructionCount'])
            sentence = DB.db.gen_insert_sentence('ProSeqSiz', [pss[0], pss[2], whole_list])
            env.close()
            return whole_list, sentence
        except Exception as e:
            print(f'=> While env.step(), Error {e} occurs.')
            print(f'=> Error file: {pss[0]}')
            whole_list = [GLOBAL_VALUE.INF for _ in range(len(pss[2]) + 1)]
            sentence = DB.db.gen_insert_sentence('ProSeqSiz', [pss[0], pss[2], whole_list])
            env.close()
            return whole_list, sentence
        finally:
            pass

    @staticmethod
    def evaluate(pss: tuple, sql_list) -> int:
        """
        Call 'run' function
        """
        return EnvManager.evaluate_whole(pss, sql_list)[-1]

    @staticmethod
    def evaluate_whole(pss: tuple, sql_list=None) -> list[int]:
        """
        Call 'run_whole' function
        """
        size_list, _ = EnvManager.run_whole(pss, sql_list=sql_list)
        return size_list

    @staticmethod
    def multi_evaluate(pss_list: list[tuple]) -> list[int]:
        """
        Evaluate multiple program-sequence using multi-thread technique
        :param pss_list: a list of exported content of ProSeqSize
        :return:
        """
        whole = EnvManager.multi_evaluate_whole(pss_list)
        res = [ls[-1] for ls in whole]
        return res

    def evaluate_and_get_min(self, pss: tuple, sql_list):
        """
        Evaluate multiple program-sequence using multi-thread technique.
        Similar to 'multi_evaluate_whole'.
        But extra return new sequence,
        About new sequence: if s[:m] perform not worse than s[:], return s[:m] with smallest m
        This was a trick used in [Learning Compiler Pass Orders using Coreset and Normalized Value Prediction].
        which can reduce the length of pass sequence.
        In this work, we use delta-debugging technique rather than this naive trick.
        """
        db_res = DB.db.select('ProSeqSiz',
                              ['size_list'],
                              ['program', 'sequence'],
                              [pss[0], pss[2]])

        if db_res is not None:
            whole = str_to_int_list(db_res[0])
        else:
            whole, sentence = self.get_whole_list(pss)
            if sentence is not None:
                sql_list.append(sentence)

        mn = 0
        for idx, val in enumerate(whole):
            if val < whole[mn]:
                mn = idx
        return (pss[0], pss[1], pss[2][:mn]), whole[mn]

    @staticmethod
    def multi_evaluate_whole(pss_list: list[tuple]) -> list[list[int]]:
        """
        Evaluate multiple program-sequence using multi-thread technique.
        :param pss_list: a list of exported content of ProSeqSize
        :return: Using a python code to explain this function: 'return [evaluate_whole(pss) for pss in pss_list]'
        """
        result = [[] for _ in range(len(pss_list))]
        futures = []

        def fun(index, temp_pss: tuple) -> (int, list[int]):
            return index, EnvManager.evaluate_whole(temp_pss)

        with ThreadPoolExecutor(max_workers=EnvManager.max_concurrency) as executor:
            for idx, pro_seq in enumerate(pss_list):
                db_res = DB.db.select('ProSeqSiz',
                                      ['size_list'],
                                      ['program', 'sequence'],
                                      [pro_seq[0], pro_seq[2]])
                if db_res is not None:
                    db_res = str_to_int_list(db_res[0])
                    # print('cache hit!')
                    result[idx] = db_res
                else:
                    # print('cache miss!')
                    future = executor.submit(fun, idx, pro_seq)
                    futures.append(future)

            for future in as_completed(futures):
                idx, whole_list = future.result()
                result[idx] = whole_list
                # print('save: ', pss_list[idx])
                _ = DB.db.insert('ProSeqSiz', [pss_list[idx][0], pss_list[idx][2], whole_list])
                # print('feed_back: ', feed_back)

        return result

    @staticmethod
    def evaluate_init(pss: tuple, sql_list=None):
        db_res = DB.db.select('ProSeqSiz',
                              ['size_list'],
                              ['program', 'sequence'],
                              [pss[0], [GLOBAL_VALUE.INIT]])
        if db_res is not None:
            return str_to_int_list(db_res[0])[-1]
        env = EnvManager.make_env(pss[1])
        res = env.observation['IrInstructionCount']
        env.close()
        if sql_list is not None:
            sentence = DB.db.gen_insert_sentence('ProSeqSiz', [pss[0], [GLOBAL_VALUE.INIT], [res]])
            sql_list.append(sentence)
        else:
            DB.db.insert('ProSeqSiz', [pss[0], [GLOBAL_VALUE.INIT], [res]])
        return res

    @staticmethod
    def evaluate_O0(pro_seq: tuple):
        pass

    @staticmethod
    def evaluate_Oz(pss: tuple, sql_list=None):
        db_res = DB.db.select('ProSeqSiz',
                              ['size_list'],
                              ['program', 'sequence'],
                              [pss[0], [GLOBAL_VALUE.OZ]])
        if db_res is not None:
            return str_to_int_list(db_res[0])[-1]
        env = EnvManager.make_env(pss[1])
        res = env.observation['IrInstructionCountOz']
        env.close()
        if sql_list is not None:
            sentence = DB.db.gen_insert_sentence('ProSeqSiz', [pss[0], [GLOBAL_VALUE.OZ], [res]])
            sql_list.append(sentence)
        else:
            DB.db.insert('ProSeqSiz', [pss[0], [GLOBAL_VALUE.OZ], [res]])
        return res


if __name__ == '__main__':
    env = compiler_gym.make(
        "llvm-ic-v0",
        reward_space='IrInstructionCount',
    )
    print(env.benchmarks)
    pass
