import os
from multiprocessing import Process, Queue

from tqdm import tqdm

from Source.DataBase.DBWrapper import DBWrapper


def worker(func, input_q: Queue, output_q: Queue):
    for task_id, args in iter(input_q.get, None):
        sql_list = []
        func_res = func(*args, sql_list)
        output_q.put((task_id, func_res, sql_list))


def start_multiprocess_tasks(fun, args, worker_count=os.cpu_count(), desc='Unknown', order=True):
    n = len(args)
    sql_lists = []
    res = [None for _ in range(n)]
    task_queue = Queue()
    done_queue = Queue()
    for _ in range(worker_count):
        Process(target=worker, args=(fun, task_queue, done_queue)).start()
    for arg_id, arg in enumerate(args):
        task_queue.put((arg_id, arg))
    for _ in range(worker_count):
        task_queue.put(None)
    for _ in tqdm(range(n), desc=f'Running {desc}'):
        arg_id, return_value, sql_list = done_queue.get()
        res[arg_id] = return_value
        sql_lists.extend(sql_list)
    DBWrapper.execute_sql_list(sql_lists)
    return res
