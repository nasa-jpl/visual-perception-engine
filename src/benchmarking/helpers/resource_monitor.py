import csv
import os
import time
import json
import multiprocessing
import torch
import pycuda.driver as cuda
from contextlib import contextmanager
from jtop import jtop, JtopException


def B2GB(n_bytes, round_3d=False):
    return round(n_bytes / 1024**3, 3) if round_3d else n_bytes / 1024**3


def MB2GB(n_bytes, round_3d=False):
    return round(n_bytes / 1024**2, 3) if round_3d else n_bytes / 1024**2


def GB2B(n_gb):
    return n_gb * 1024**3


@contextmanager
def Duration():
    # measures duration of execution of wrapped routine
    start = time.perf_counter()
    try:
        yield None
    finally:
        print("Total time:", time.perf_counter() - start)


def monitor_gpu_memory(rounded=False):
    # Get the current free and total memory on the GPU
    free, total = cuda.mem_get_info()
    # Calculate the used memory
    used = total - free
    return B2GB(used, rounded), B2GB(total, rounded)


@contextmanager
def monitor_jtop_process(event, log_dir=""):
    filepath = os.path.join(log_dir, "resource_usage.csv")
    try:
        with jtop(
            interval=0.5
        ) as jetson:  # smallest interval recommended by https://github.com/rbonghi/jetson_stats/issues/414
            # Make csv file and setup csv
            with open(filepath, "w") as csvfile:
                stats = jetson.stats
                # Initialize cws writer ["gpu load", "gpu memory", "used_ram"]
                stat_keys = [key for key in stats.keys() if key in [f"CPU{i}" for i in range(1, 13)]]
                writer = csv.DictWriter(csvfile, fieldnames=stat_keys + ["gpu_load", "gpu_memory", "used_ram"])
                # Write header
                writer.writeheader()
                # Start loop
                while jetson.ok() and not event.is_set():
                    stats = {k: v for (k, v) in jetson.stats.items() if k in stat_keys}
                    # Write row
                    stats.update(
                        {
                            "gpu_load": jetson.gpu["gpu"]["status"]["load"],
                            "gpu_memory": MB2GB(jetson.memory["RAM"]["shared"]),
                            "used_ram": MB2GB(jetson.memory["RAM"]["used"]),
                        }
                    )
                    writer.writerow(stats)
    except JtopException as e:
        print(f"Error occured during resource monitoring: {e}")
    except KeyboardInterrupt:
        print("Resource monitoring was closed with CTRL-C")
    except IOError as e:
        print("I/O error occured during resource monitoring: ", e)


@contextmanager
def ResourceMonitoring(log_dir=""):
    """
    Context manager to run jtop in a separate process to monitor system resources.

    :param output_file: File path to store monitoring data.
    :param duration: Duration in seconds to run the monitoring.
    """
    # create directory if not
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Create the process for monitoring
    e = multiprocessing.Event()
    process = multiprocessing.Process(target=monitor_jtop_process, args=(e, log_dir))
    process.start()

    # Yield control back to the caller
    metrics = {
        "inference_times": [],
        "total_processing_times": [],
    }
    try:
        yield metrics
    finally:
        e.set()
        process.join()
        process.terminate()

        # save extra metrics
        with open(os.path.join(log_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        print("Monitoring process finished.")


def cpu_intesive_function(target_time=10):
    # target_time: desired approximate time of execution in seconds
    assert target_time <= 20, "Too large values can crash the system"
    large_number_10s = 0.98e8
    number = large_number_10s * (target_time / 10)
    _ = sum([x * x for x in range(int(number))])


def gpu_intensive_function(target_memory=1):
    # target_memory: approximate amount of GPU memory used
    assert target_memory <= 40, "For safety test memory of less than 40GB"

    from math import sqrt

    side_size = sqrt(GB2B(target_memory) / torch.float32.itemsize / 3)
    assert (side_size**2) * 3 * B2GB(torch.float32.itemsize) == target_memory
    side_size = int(side_size)

    mat1 = torch.rand(size=(side_size, side_size), device=torch.device("cuda"), dtype=torch.float32)
    mat2 = torch.rand(size=(side_size, side_size), device=torch.device("cuda"), dtype=torch.float32)
    _ = mat1 @ mat2


if __name__ == "__main__":
    # main function is supposed to work as a unit test for resource monitoring

    with ResourceMonitoring("runs/tests".format(**os.environ)):
        cpu_intesive_function()
        gpu_intensive_function()
