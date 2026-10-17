import os
import time
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import asyncio
from typing import List, Dict, Any, Optional

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
    ('status', pa.int64())
])

class RequestLogger:
    """
    Logs every request together with the server state to a Parquet file.
    Implements Task 5 of Phase 2.
    """
    def __init__(self, log_dir: str = "logs", run_id: str = "run_0", buffer_size: int = 50, flush_interval: float = 2.0):
        self.log_dir = log_dir
        self.run_id = run_id
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self.buffer: List[Dict[str, Any]] = []
        self._running = False
        self._lock = asyncio.Lock()
        self._writer: Optional[pq.ParquetWriter] = None
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.file_path = os.path.join(self.log_dir, f"requests_{self.run_id}_{int(time.time())}.parquet")
        
    async def start(self):
        self._running = True
        self._writer = pq.ParquetWriter(self.file_path, SCHEMA)
        asyncio.create_task(self._flush_loop())
        
    async def stop(self):
        self._running = False
        await self.flush()
        if self._writer:
            self._writer.close()
            self._writer = None

    async def log(self, entry: Dict[str, Any]):
        # Ensure default values match schema
        entry['run_id'] = self.run_id
        async with self._lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.buffer_size:
                await self._flush_unlocked()

    async def flush(self):
        async with self._lock:
            await self._flush_unlocked()

    async def _flush_unlocked(self):
        if not self.buffer or not self._writer:
            return
            
        df = pd.DataFrame(self.buffer)
        # Ensure column order and types match schema
        table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
        self._writer.write_table(table)
        self.buffer.clear()

    async def _flush_loop(self):
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self.flush()
