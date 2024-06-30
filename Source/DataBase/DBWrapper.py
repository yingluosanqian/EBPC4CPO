import sys
sys.path.append('./../../')

import os
import sys

from tqdm import tqdm

sys.path.append('../../')

import sqlite3
import pandas as pd

from Source.Util.util import str_list_to_string, get_root_path, db_filter


class DBWrapper:
    def __init__(self, db_name: str):
        self.db_name = db_name
        db_path = f'{get_root_path()}/DataBase/{self.db_name}.db'
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self.init_table()

    def init_table(self):
        if self.db_name == 'ProFea':
            self.create_table('ProFea',
                              ['program', 'feature_type', 'feature_value'],
                              ['TEXT', 'TEXT', 'TEXT'],
                              ['program', 'feature_type'])
        elif self.db_name == 'ProSeqSiz':
            self.create_table('ProSeqSiz',
                            ['program', 'sequence', 'size_list'],
                            ['TEXT', 'TEXT', 'TEXT'],
                            ['program', 'sequence'])

    def __del__(self):
        self.connection.commit()
        self.connection.close()

    def create_table(self, table_name: str, args: list[str], types: list[str], primary_keys: list[str]):
        sentence = f'CREATE TABLE IF NOT EXISTS {table_name} (\n'
        for arg, type in zip(args, types):
            sentence += f'    {arg} {type},\n'
        str_primary_keys = str_list_to_string(primary_keys, ', ')
        sentence += f'    PRIMARY KEY ({str_primary_keys})\n'
        sentence += ')\n'
        return self.execute_and_commit((sentence, self.db_name))

    def execute_and_commit(self, sentence):
        sentence, db_name = sentence
        res = self.cursor.execute(sentence).fetchall()
        self.connection.commit()
        return res

    # def execute_from_lists(self, sql_lists):
    #     for sql_list in tqdm(sql_lists, desc='Writing to Database', dynamic_ncols=True):
    #         self.execute_from_lists(sql_list)

    def execute_from_list(self, sql_list):
        for sentence in tqdm(sql_list, desc='Writing to Database', dynamic_ncols=True):
            self.cursor.execute(sentence)
        self.connection.commit()

    @staticmethod
    def execute_sql_list(sql_list):
        if sql_list is None or sql_list == []:
            return
        else:
            _, db_name = sql_list[0]
            if db_name == 'ProSeqSiz' in sql_list[0]:
                cor_db = DB.db
            elif db_name == 'ProFea' in sql_list[0]:
                cor_db = DB.feature_db
            else:
                raise Exception(f'Unregister database {db_name}')
            for sentence, _ in tqdm(sql_list, desc='Writing to Database', dynamic_ncols=True):
                cor_db.cursor.execute(sentence)
            cor_db.connection.commit()

    def gen_insert_sentence(self, table_name: str, values: list, strategy='OR IGNORE'):
        sentence = f'INSERT {strategy} INTO {table_name} VALUES (\n'
        str_values = ', '.join([str(db_filter(value)) for value in values])
        sentence += f'    {str_values}\n'
        sentence += ')\n'
        return sentence, self.db_name

    def insert(self, table_name: str, values: list, strategy='OR IGNORE'):
        sentence = self.gen_insert_sentence(table_name, values, strategy)
        # print(f'[insert]\n{sentence}')
        return self.execute_and_commit(sentence)

    def select(self, table_name: str, col: list[str], primary_keys: list[str], values: list[str]):
        str_col = str_list_to_string(col, ', ')
        sentence = f'SELECT {str_col}\n'
        sentence += f'FROM {table_name}\n'
        kv_pairs = [f"{key} = {db_filter(value)}" for key, value in zip(primary_keys, values)]
        sentence += f'WHERE {str_list_to_string(kv_pairs, " AND ")};\n'
        # print(f'[select]\n{sentence}')
        res = self.cursor.execute(sentence).fetchone()
        self.connection.commit()
        return res

    def select_all(self, table_name: str):
        sentence = f'SELECT * FROM {table_name}'
        return self.cursor.execute(sentence).fetchall()

    def delete_table_if_exist(self, table_name: str):
        sentence = f'DROP TABLE IF EXISTS {table_name};'
        res = self.cursor.execute(sentence).fetchall()
        self.connection.commit()
        return res

    def exist_table(self, table_name: str):
        sentence = f"SELECT * FROM sqlite_master WHERE type='table' AND name={db_filter(table_name)};"
        res = self.cursor.execute(sentence).fetchone() is not None
        self.connection.commit()
        return res

    def to_excel(self, name):
        # 从数据库读取数据到DataFrame
        df = pd.read_sql_query(f"SELECT * FROM {name}", self.connection)

        # 将DataFrame保存为Excel文件
        df.to_excel('output.xlsx', index=False)

    def QAQ(self):
        sentence = f"DELETE FROM ProSeqSiz WHERE sequence = {db_filter([-1])};"
        self.cursor.execute(sentence).fetchone()
        self.connection.commit()
        # with open(f'{get_root_path()}/Train/Benchmark/codeforces_test.txt') as f:
        #     for line in f:
        #         name = line.strip()
        #         sentence = f"SELECT * FROM ProSeqSiz WHERE program = {db_filter(name)};"
        #         res = self.cursor.execute(sentence).fetchone()
        #         self.connection.commit()
        #         print(res)
        # with open(f'{get_root_path()}/Train/Benchmark/codeforces_test.txt') as f:
        #     for line in f:
        #         name = line.strip()
        #         sentence = f"DELETE FROM ProSeqSiz WHERE program = {db_filter(name)};"
        #         print(sentence)
        #         self.connection.commit()


class DB:
    db = DBWrapper('ProSeqSiz')
    feature_db = DBWrapper('ProFea')


if __name__ == '__main__':
    db = DBWrapper('ProSeqSiz')
    # db.QAQ()
    # db.to_excel('ProSeqSiz')
    # print(db.delete_table_if_exist(f'ProSeqSiz'))
    #
    for i in range(10):
        print(db.delete_table_if_exist(f'__it_train_base_{i}_greedy'))
        print(db.delete_table_if_exist(f'__it_train_base_{i}_spread'))
        print(db.delete_table_if_exist(f'__it_train_base_{i}_probdd'))
        print(db.delete_table_if_exist(f'__it_train_base_{i}_boost'))
        print(db.delete_table_if_exist(f'__it_train_no_trimming_{i}_greedy'))
        print(db.delete_table_if_exist(f'__it_train_no_trimming_{i}_spread'))
        print(db.delete_table_if_exist(f'__it_train_no_trimming_{i}_probdd'))
        print(db.delete_table_if_exist(f'__it_train_no_trimming_{i}_boost'))
        print(db.delete_table_if_exist(f'__it_train_no_sharing_{i}_greedy'))
        print(db.delete_table_if_exist(f'__it_train_no_sharing_{i}_spread'))
        print(db.delete_table_if_exist(f'__it_train_no_sharing_{i}_probdd'))
        print(db.delete_table_if_exist(f'__it_train_no_sharing_{i}_boost'))
    # db.create_table('ProSeqSiz',
    #                 ['program', 'sequence', 'size_list'],
    #                 ['TEXT', 'TEXT', 'TEXT'],
    #                 ['program', 'sequence'])yao
    #

    # print(db.exist_table('greedy_test_x'))
