import asyncio
import json
from pathlib import Path
import statistics
import sys
import time

from httpcore._synchronization import current_async_library
from atfm_experiments.load.runtime import provenance

async def main():
    for _ in range(100):
        assert current_async_library() == 'asyncio'
    trials = []
    for _ in range(7):
        start = time.perf_counter()
        for _ in range(10000):
            current_async_library()
        trials.append(time.perf_counter() - start)
    result = {'iterations': 10000, 'warmup': 100, 'seconds': trials,
              'median_s': statistics.median(trials), 'environment': provenance()}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
    print(sys.argv[1], result['median_s'])

asyncio.run(main())
