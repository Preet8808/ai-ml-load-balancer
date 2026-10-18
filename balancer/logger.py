import os
import time
import asyncio
from typing import List, Dict, Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ('timestamp', pa.float64()),
    ('run_id', pa.string()),
    ('req_type', pa.string()),
    ('payload_kb', pa.float64()),
    ('server_id', pa.string()),
    ('srv_cpu_pct', pa.float64()),
    ('srv_mem_pct', pa.float64()),
    ('inflight', pa.int64()),
    ('req_rate_5s', pa.float64()),
    ('rt_ewma_ms', pa.float64()),
    ('rt_p95_5s_ms', pa.float64()),
    ('err_rate_5s', pa.float64()),
    ('cpu_limit', pa.float64()),
    ('cpu_delta_5s', pa.float64()),
    ('response_ms', pa.float64()),
    ('status', pa.int64()),
])


class RequestLogger:
    """
    Logs every request together with the server state to a Parquet file.

    A Parquet file only becomes readable once its footer is written, which
    happens when the writer is closed. Rotating the writer per run (and
    sealing it explicitly through /admin/run) means each run produces one
    complete, independently readable file with a real `run_id`.

    Writing continuously to a single writer for the lifetime of the process
    produced corrupt copies *and* made every row share one run_id, which
    silently breaks run-grouped cross-validation.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        run_id: str = "run_0",
        buffer_size: int = 50,
        flush_interval: float = 2.0,
    ):
        self.log_dir = log_dir
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer: List[Dict[str, Any]] = []

        self._running = False
        self._lock = asyncio.Lock()
        self._writer: Optional[pq.ParquetWriter] = None
        self._file_path: Optional[str] = None
        self._run_id: str = run_id
        self._rows: int = 0
        self._dropped: int = 0
        self._sealed: List[Dict[str, Any]] = []
        self._flush_task: Optional[asyncio.Task] = None

        os.makedirs(self.log_dir, exist_ok=True)

    # -- lifecycle ---------------------------------------------------------

    async def start(self):
        """
        Start the flush loop. No file is opened until `start_run` is called, so
        requests arriving outside a run are dropped and counted rather than
        being mislabelled into whatever run happens to be current.
        """
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self):
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except (asyncio.CancelledError, Exception):
                pass
            self._flush_task = None
        await self.seal()

    # -- run rotation ------------------------------------------------------

    async def start_run(self, run_id: str) -> Dict[str, Any]:
        """Seal the current file (if any) and start a new one for `run_id`."""
        async with self._lock:
            sealed = await self._seal_unlocked()
            self._run_id = run_id
            self._file_path = os.path.join(
                self.log_dir, f"{run_id}_{int(time.time() * 1000)}.parquet"
            )
            self._writer = pq.ParquetWriter(self._file_path, SCHEMA)
            self._rows = 0
            return {"previous": sealed, "current": self._file_path, "run_id": run_id}

    async def seal(self) -> Dict[str, Any]:
        """Flush and close the current file so it is readable on disk."""
        async with self._lock:
            return await self._seal_unlocked()

    async def _seal_unlocked(self) -> Optional[Dict[str, Any]]:
        if self._writer is None:
            return None
        await self._flush_unlocked()
        self._writer.close()
        info = {
            "path": self._file_path,
            "run_id": self._run_id,
            "rows": self._rows,
            "bytes": os.path.getsize(self._file_path) if self._file_path else 0,
        }
        self._sealed.append(info)
        self._writer = None
        self._file_path = None
        return info

    # -- logging -----------------------------------------------------------

    async def log(self, entry: Dict[str, Any]):
        async with self._lock:
            if self._writer is None:
                self._dropped += 1
                return
            entry['run_id'] = self._run_id
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                await self._flush_unlocked()

    async def flush(self):
        async with self._lock:
            await self._flush_unlocked()

    async def _flush_unlocked(self):
        if not self.buffer or self._writer is None:
            return

        df = pd.DataFrame(self.buffer)
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        self._writer.write_table(table)
        self._rows += len(self.buffer)
        self.buffer.clear()

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(self.flush_interval)
            try:
                await self.flush()
            except Exception:
                # Never let a transient write error kill the flush loop
                pass

    # -- introspection -----------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "run_id": self._run_id,
            "current_file": self._file_path,
            "rows_written": self._rows,
            "buffered": len(self.buffer),
            "dropped": self._dropped,
            "sealed_runs": self._sealed,
        }
