from glob import glob
import json
from dataclasses import dataclass
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid

import click
import pyarrow.parquet as pq
import duckdb
from tqdm import tqdm

from kiwipiepy import Kiwi
import pyarrow as pa
from multiprocessing import Pool, cpu_count

from multiprocessing import Process, Queue, cpu_count  # add Queue, Process
from typing import Dict, Tuple, List
from collections import defaultdict
from typing import List, Optional
import duckdb
import pandas as pd  # just for typing clarity, not strictly required
from tqdm import tqdm


def _consumer_worker(db_path_str: str, export_dir_str: str, job_queue: "Queue"):
    """
    Consumer: repeatedly pulls jobs from job_queue and writes files to disk
    using _export_one_job. Stops when it receives a sentinel (None).
    """
    while True:
        job = job_queue.get()
        if job is None:
            break
        _export_one_job((db_path_str, export_dir_str, job))


def _export_one_job(args):
    """
    Worker: export a single (audio, transcript, metadata) pair.

    args: (db_path_str, export_dir_str, job_dict)
    job_dict keys:
        uid, orig_uid, duration_sec, target_index, target_text,
        candidate_text, bm25_score, score_overlap,
        source_parquet, source_row_number
    """
    db_path_str, export_dir_str, job = args
    export_dir = Path(export_dir_str)
    uid = job["uid"]
    orig_uid = job["oring_uid"]
    source_parquet = job["source_parquet"]
    source_row_number = job["source_row_number"]

    # Re-open DuckDB in this worker just to read audio from parquet
    conn = duckdb.connect(db_path_str, read_only=True)
    try:
        try:
            audio_row = conn.execute(
                """
                SELECT audio
                FROM read_parquet(
                    ?, 
                    union_by_name   = true,
                    filename        = true,
                    file_row_number = true
                )
                WHERE file_row_number = ?
                """,
                [str(source_parquet), int(source_row_number)],
            ).fetchdone()
        except Exception:
            audio_row = conn.execute(
                """
                SELECT audio_wav_bytes
                FROM read_parquet(
                    ?, 
                    union_by_name   = true,
                    filename        = true,
                    file_row_number = true
                )
                WHERE file_row_number = ?
                """,
                [str(source_parquet), int(source_row_number)],
            ).fetchone()
    finally:
        conn.close()
