import threading
from queue import Empty
from time import perf_counter

from cuda import cuda

from vp_engine.cuda_utils import checkCudaErrors
from utils.logging_utils import MESSAGE


class StoppableThread(threading.Thread):
    """Thread class with a terminate() method. The thread itself has to check
    regularly for the terminate() condition."""

    def __init__(self, *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


class QueueReceiverThread(StoppableThread):
    def __init__(self, queue, output_list, n_inputs, head_name, logger, timeout: int = 10, *args, **kwargs):
        super(QueueReceiverThread, self).__init__(*args, **kwargs)
        self.queue = queue
        self.output_list = output_list
        self.n_inputs = n_inputs
        self.head_name = head_name
        self.logger = logger
        self.timeout = timeout  # amount of time to wait for the next output

    def run(self):
        checkCudaErrors(cuda.cuInit(0))
        cu_device = checkCudaErrors(cuda.cuDeviceGet(0))
        cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cu_device))
        last_received_t = perf_counter()
        while True:
            try:
                n, output = self.queue.get_nowait()
                self.logger.info(MESSAGE.OUTPUT_RECEIVED.format(n=n, head_name=self.head_name))
                last_received_t = perf_counter()
                output = {k: v.clone() for k, v in output.items()}
                self.output_list.append((n, output))
                self.logger.info(MESSAGE.OUTPUT_CLONED.format(n=n, head_name=self.head_name))
            except Empty:
                pass
            finally:
                if perf_counter() - last_received_t > self.timeout:
                    self.logger.warning(
                        f"Timeout of {self.timeout} seconds reached. Stopping the thread for {self.head_name} head."
                    )
                    break

            # stop when finished or if it takes too much time
            if len(self.output_list) == self.n_inputs:
                break

        self.logger.info(
            f"Thread for {self.head_name} head is stopping. Received {len(self.output_list)} / {self.n_inputs} outputs ({100 * len(self.output_list) / self.n_inputs:.1f}%)"
        )
