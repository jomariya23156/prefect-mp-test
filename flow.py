import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
from typing import Union, List, Dict, Optional, Any
from prefect import flow, task, get_run_logger
from tasks import (split_ids, load_files, process_chunk)
from utils import say_debug

from prefect_dask import DaskTaskRunner

# use 60% of all processors
NUM_PROCESSES = max(1, int(os.cpu_count() * 0.6))
print(f"Setting up a Dask cluster {NUM_PROCESSES} processes")

dask_runner = DaskTaskRunner(
    cluster_class="dask.distributed.LocalCluster",
    cluster_kwargs={
        "n_workers": NUM_PROCESSES,
        "threads_per_worker": 2,
        "host": "0.0.0.0",
        "dashboard_address": ":8505",
    },
)
print('Initialized dask runner')


# main execution logic
@task(name="parralel_execute", log_prints=True)
def parralel_execute(base_file_path: str, input_ids: List[int]):
    logger = get_run_logger()
    say_debug("Inside parralel function")

    ## Load all required secrets and vars
    logger.info("Loading files...")
    loaded_text = load_files(
        base_file_path
    )
    logger.info(f"Loaded text: {loaded_text}")

    n_results = process_chunk(input_ids, loaded_text)

    logger.info(f"N results: {n_results}")

    return n_results


@flow(name="main_execute_flow", task_runner=dask_runner, log_prints=True)
def main_execute_flow(input_ids: List[int]):
    logger = get_run_logger()
    say_debug("Inside main flow")

    ids_chunks = split_ids(input_ids, NUM_PROCESSES)
    logger.info(f"Split work_ids!")

    futures = []
    for idx, chunk in enumerate(ids_chunks):
        future = parralel_execute.submit(base_file_path=".", input_ids=chunk)
        futures.append(future)
        logger.info(f"Submitted chunk no. {idx}")

    logger.info("Waiting all processes to finish...")
    results = []
    # make sure all tasks are done before terminating the script
    for future in tqdm(futures, total=len(futures)):
        try:
            results.append(future.result())
        except Exception as e:
            print(f"Error processing chunk: {e}")
    logger.info(
        f"Total ids processed: {sum(results)} | From all upstream ids: {len(input_ids)}"
    )


if __name__ == "__main__":
    input_ids = list(range(100))
    main_execute_flow(input_ids)
