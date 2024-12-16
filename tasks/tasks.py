import os
import numpy as np
from typing import Union, List, Dict, Optional, Any
from prefect import task, get_run_logger

@task(name="split_ids")
def split_ids(input_ids: List[str], n_chunks: int) -> List[List[str]]:
    logger = get_run_logger()
    logger.info(f"Splitting input_ids into {n_chunks} chunks")
    input_ids_chunks = np.array_split(input_ids, n_chunks)
    # convert back to native list so it's easier to pass between process
    input_ids_chunks = [chunk.tolist() for chunk in input_ids_chunks]
    return input_ids_chunks

@task(name="load_files")
def load_files(base_file_path="."):
    logger = get_run_logger()
    with open(os.path.join(base_file_path, 'empty.txt'), 'r') as f:
        txt = f.read()
    logger.info('Loaded .txt')
    return txt

@task(name="process_chunk")
def process_chunk(chunk_ids: List[int], text: str):
    logger = get_run_logger()
    logger.info(f"Process got: {text}")

    count = 0
    for id_name in chunk_ids:
        count += 1
    return count