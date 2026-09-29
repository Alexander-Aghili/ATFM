"""Own the benchmark client's process group, including its worker processes."""
import os
import signal
import subprocess


def run_client(args, log, timeout=180):
    with subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
        try:
            status = process.wait(timeout=timeout)
            if status:
                raise subprocess.CalledProcessError(status, args)
        finally:
            stop_group(process)


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
