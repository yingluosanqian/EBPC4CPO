import os

from Source.Util.multitask import start_multiprocess_tasks
from Source.Util.raytool import start_ray_tasks
from Source.Util.util import GLOBAL_VALUE


def start_tasks(fun, tasks, worker_count=os.cpu_count(), order=True, desc='Unknown'):
    if GLOBAL_VALUE.PARALLEL_STRATEGY == GLOBAL_VALUE.RAY:
        return start_ray_tasks(fun, tasks, worker_count=worker_count, desc=desc, order=order)
    elif GLOBAL_VALUE.PARALLEL_STRATEGY == GLOBAL_VALUE.MULTIPROCESSING:
        return start_multiprocess_tasks(fun, tasks, worker_count=worker_count, desc=desc, order=order)
    else:
        raise Exception('Unknown Parallel Strategy')
