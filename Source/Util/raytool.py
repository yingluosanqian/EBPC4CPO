import os

import ray

from tqdm import tqdm

from Source.DataBase.DBWrapper import DB, DBWrapper


@ray.remote
def worker(fun, args):
    sql_list = []
    res = fun(*args, sql_list)
    return res, sql_list


def start_ray_tasks(fun, args_list, worker_count=os.cpu_count(), desc='Unknown', order=True):
    res, all_sql_list = start_ray_tasks_with_sql(fun, args_list, worker_count=worker_count, desc=desc, order=order)

    total = []
    for sql_list in all_sql_list:
        total.extend(sql_list)
    DBWrapper.execute_sql_list(total)

    return res


def start_ray_tasks_with_sql(fun, args_list, worker_count=os.cpu_count(), desc='Unknown', order=True):
    n = len(args_list)
    tasks = [worker.remote(fun, args) for args in tqdm(args_list, total=n, desc=f'{desc} [Loading]')]

    if order:
        with tqdm(total=n, desc=f'{desc} [Running]') as bar:
            temp_tasks = tasks
            while temp_tasks:
                done_ids, temp_tasks = ray.wait(temp_tasks, num_returns=1)
                bar.update(1)

        res, all_sql_list = zip(*ray.get(tasks))
        res = list(res)
        all_sql_list = list(all_sql_list)
        return res, all_sql_list
    else:
        res, all_sql_list = [], []
        with tqdm(total=n, desc=f'{desc} [Running]') as bar:
            temp_tasks = tasks
            while temp_tasks:
                done_ids, temp_tasks = ray.wait(temp_tasks, num_returns=1)
                temp1, temp2 = ray.get(done_ids[0])
                res.append(temp1)
                all_sql_list.append(temp2)
                bar.update(1)

        return res, all_sql_list


# def one_by_one_worker(fun, args):
#     sql_list = []
#     res = fun(*args, sql_list)
#     return res, sql_list
#
#
# def start_one_by_one_tasks(fun, args_list, desc='Unknown', order=True):
#     res, all_sql_list = zip(*[one_by_one_worker(fun, args) for args in tqdm(args_list, desc=f'{desc} [Running]')])
#     res = list(res)
#     all_sql_list = list(all_sql_list)
#
#     total = []
#     for sql_list in all_sql_list:
#         total.extend(sql_list)
#     DB.db.execute_from_list(total)
#
#     return res
