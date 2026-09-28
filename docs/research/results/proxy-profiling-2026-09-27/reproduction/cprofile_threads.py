import cProfile
import json
import pstats
import sys
import threading

def worker_only():
    return sum(range(10000))

profiler = cProfile.Profile()
profiler.enable()
worker = threading.Thread(target=worker_only)
worker.start()
worker.join()
profiler.disable()
stats = pstats.Stats(profiler)
print(json.dumps({'python': sys.version, 'worker_captured': any(k[2] == 'worker_only' for k in stats.stats)}))
