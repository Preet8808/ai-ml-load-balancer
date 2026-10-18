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


def _file_size(path: Optional[str]) -> int:
    """
    Size of a log file, or 0 if it has already been moved or removed.

    Sealing must not fail just because something else already picked the file
    up; the previous behaviour raised FileNotFoundError and took the whole
    collection run down with it.
    """
    if not path:
        return 0
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


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

    The request path never blocks. `log()` only appends to a list, which is
    atomic inside a single-threaded event loop, and the Parquet write happens
    in a worker thread. Holding a lock across the write made every proxied
    request queue behind the flush: with ~200 concurrent requests and a
    20 ms write to the bind mount, `/ping` measured 2.6 s of pure logging
    latency while the backends were answering in 10 ms.
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

        self._running = False
        self._writer: Optional[pq.ParquetWriter] = None
        self._file_path: Optional[str] = None
        self._run_id: str = run_id
        self._rows: int = 0
        self._dropped: int = 0
        self._sealed: List[Dict[str, Any]] = []
        self._flush_task: Optional[asyncio.Task] = None
        self._buffer: List[Dict[str, Any]] = []
        # Serialises writers between the flush task and seal()/start_run()
        self._io_lock = asyncio.Lock()
        # Lets stop() wake the flush loop instead of waiting out its interval,
        # and lets log() ask for a drain once the buffer is worth writing
        self._wake = asyncio.Event()

        os.makedirs(self.log_dir, exist_ok=True)

    @property
    def buffer(self) -> List[Dict[str, Any]]:
        """Rows waiting to be written (exposed for tests)."""
        return self._buffer

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
        """
        Stop the flush loop and close the current file.

        The loop is woken and awaited rather than cancelled: cancelling it
        while a worker thread was mid-write let the thread race with the
        `close()` in `seal()` and produced unreadable Parquet files.
        """
        self._running = False
        self._wake.set()

        if self._flush_task:
            try:
                await self._flush_task
            except Exception:
                pass
            self._flush_task = None
        await self.seal()

    # -- run rotation ------------------------------------------------------

    async def start_run(self, run_id: str) -> Dict[str, Any]:
        """Seal the current file (if any) and start a new one for `run_id`."""
        sealed = await self.seal()
        self._run_id = run_id
        self._file_path = os.path.join(
            self.log_dir, f"{run_id}_{int(time.time() * 1000)}.parquet"
        )
        self._writer = pq.ParquetWriter(self._file_path, SCHEMA)
        self._rows = 0
        return {"previous": sealed, "current": self._file_path, "run_id": run_id}

    async def seal(self) -> Optional[Dict[str, Any]]:
        """
        Flush and close the current file so it is readable on disk.

        This must never raise. Sealing happens between load runs, and a failure
        here previously wedged the logger: the writer stayed installed while
        its file had already been removed from the log directory, so every
        subsequent /admin/run returned 500 and the whole collection aborted
        after the first run.
        """
        async with self._io_lock:
            if self._writer is None:
                return None

            path, run_id = self._file_path, self._run_id
            error: Optional[str] = None

            try:
                await self._drain()
            except Exception as exc:
                error = f"flush failed: {exc}"
                # Rows that could not be written belong to this run. Keeping
                # them would attribute them to the *next* run_id, which is
                # exactly the mislabelling this logger exists to prevent.
                self._dropped += len(self._buffer)
                self._buffer = []

            try:
                self._writer.close()
            except Exception as exc:
                error = error or f"close failed: {exc}"

            # Clear the writer whatever happened, so the next run can start and
            # the failure surfaces as a reported `error` instead of a 500.
            self._writer = None
            self._file_path = None

            info = {
                "path": path,
                "run_id": run_id,
                "rows": self._rows,
                "bytes": _file_size(path),
                "error": error,
            }
            self._sealed.append(info)
            return info

    # -- logging -----------------------------------------------------------

    async def log(self, entry: Dict[str, Any]):
        """
        Record one request. Returns immediately.

        Kept async so callers can `await` it, but it performs no I/O and takes
        no lock, so it cannot delay the proxy. Once enough rows have piled up
        it only *signals* the flush loop, rather than writing inline.
        """
        if self._writer is None:
            self._dropped += 1
            return
        entry['run_id'] = self._run_id
        self._buffer.append(entry)
        if len(self._buffer) >= self.buffer_size:
            self._wake.set()

    async def flush(self):
        """Write buffered rows to disk."""
        async with self._io_lock:
            await self._drain()

    async def _drain(self):
        """Move buffered rows into the Parquet file, in a worker thread."""
        if not self._buffer or self._writer is None:
            return
        batch = self._buffer
        self._buffer = []
        try:
            await asyncio.to_thread(self._write_batch, batch)
        except Exception:
            self._dropped += len(batch)
            raise

    def _write_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Synchronous Parquet write. Runs off the event loop."""
        df = pd.DataFrame(batch)
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        self._writer.write_table(table)
        self._rows += len(batch)

    async def _flush_loop(self):
        """
        Drain the buffer periodically, and whenever `log()` asks for it.

        The loop only exits after any in-flight write has completed, so
        `seal()` can never close the writer underneath a worker thread.
        """
        while self._running:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.flush_interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()

            if not self._running:
                break

            try:
                async with self._io_lock:
                    await self._drain()
            except Exception:
                # Never let a transient write error kill the flush loop
                pass

    # -- introspection -----------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return {
            "run_id": self._run_id,
            "current_file": self._file_path,
            "rows_written": self._rows,
            "buffered": len(self._buffer),
            "dropped": self._dropped,
            "sealed_runs": self._sealed,
        }
