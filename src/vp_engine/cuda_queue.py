import os
import socket
from abc import ABC, abstractmethod
from typing import Literal
from queue import Full, Empty

import torch
import multiprocessing as mp
from cuda import cuda

from vp_engine.cuda_utils import CUDASharedMemorySlotWithID, checkCudaErrors
from utils.logging_utils import create_logger


class FIFOQueueInterface(ABC):
    @abstractmethod
    def put_nowait(self, item: dict[str, torch.Tensor], item_id: int) -> None:
        """Immediately put item in the queue. Should raise an exception if the queue is full."""
        pass

    @abstractmethod
    def get_nowait(self) -> tuple[int, torch.Tensor]:
        """Immediately get the oldest item from the queue. Should raise an exception if the queue is empty."""
        pass

    def put(self, item: dict[str, torch.Tensor], item_id: int) -> None:
        raise NotImplementedError("This method is not implemented. Use put_nowait instead.")

    def get(self) -> torch.Tensor:
        raise NotImplementedError("This method is not implemented. Use get_nowait instead.")

    def send_shareable_handles(self, pid: int, close_sockets: bool = True):
        """This function is here simply to match the API of CUDAQueue. It does nothing."""
        pass

    def recv_shareable_handles(self, close_sockets: bool = True):
        """This function is here simply to match the API of CUDAQueue. It does nothing."""
        pass


class CUDAQueue(FIFOQueueInterface):
    """A FIFO queue that stores tensors in shared memory.
    This queue is intented to be used in one-to-one manner, i.e., one producer and one consumer.
    Otherwise the behavior might be unexpected."""

    def __init__(
        self,
        max_size: int,
        dtype: torch.dtype,
        input_device: Literal["cpu", "cuda"],
        output_device: Literal["cpu", "cuda"],
        data_signature: dict[str, tuple[int]] | tuple[int],
        reading_synchronised: bool = True,  # set to false if after reading all operations are performed in the same stream
        is_original_process: bool = True,
        device_id: int = 0,
        add_batch_dim: bool = False,
    ):
        self.max_size = max_size
        self.dtype = dtype
        self.input_device = input_device
        self.output_device = output_device
        self.data_signature = data_signature
        assert all([None not in v for v in data_signature.values()]), (
            "None (wildcard shape) is not allowed in this data structure"
        )
        self.is_original_process = is_original_process
        self.reading_synchronised = reading_synchronised
        self.device_id = device_id
        self.add_batch_dim = add_batch_dim
        self.logger = mp.get_logger()

        self._queue = [
            CUDASharedMemorySlotWithID(
                data_signature, dtype, is_original_process, device_id=device_id, add_batch_dim=add_batch_dim
            )
            for _ in range(max_size)
        ]

        self._output_slot = self._queue[0].get_non_shared_empty_memory_slot(
            self.output_device
        )  # memory slot where the output will be written, initialized in the child processes

        self._position_locks = [mp.Lock() for _ in range(max_size)]
        self._front_ind = mp.Value("i", 0)
        self._current_size = mp.Value("i", 0)

        parent_sock, child_sock = socket.socketpair()
        os.set_inheritable(parent_sock.fileno(), True)
        os.set_inheritable(child_sock.fileno(), True)
        self._parent_sock = parent_sock
        self._child_sock = child_sock

    def __getstate__(self):
        """Return the state of the object for pickling. Intended to be used only when passing the object to a child process.
        The new state will not contain the memory pointers, nor the parent socket.
        Furhtermore, the new state will contain a flag indicating that this is not the original process.
        """
        state = self.__dict__.copy()
        state["is_original_process"] = False
        state.pop(
            "_parent_sock"
        )  # removing parent socket is not necessary but it is a good practice to separete original and child processes
        state.pop("_output_slot")  # the output slot will be initialized in the child process
        return state

    def __setstate__(self, state):
        """Set the state of the object after unpickling. Intended to be used only when passing the object to a child process.
        The new state will contain a flag indicating that this is not the original process.
        """
        self.__dict__.update(state)
        self.is_original_process = False
        self._parent_sock = None
        self._output_slot = None

    @property
    def device(self):
        return self._queue[0].device

    def close(self):
        for slot in self._queue:
            slot.close()

        # the sockets should be closed after sending and receiving the shared memory handles
        # if the handles are not transmitted, the sockets will be closed in the destructor
        # .close() is idempotent
        self._child_sock.close()
        if self._parent_sock is not None:
            self._parent_sock.close()

    def send_shareable_handles(self, destination_pid, close_sockets: bool = True):
        """Send the handles of the shared memory slots over a local connection.
        This method should be called only from the original process."""
        transmit_error = True
        try:
            for slot in self._queue:
                slot.send_shareable_handles(self._parent_sock, destination_pid)
            transmit_error = False
        finally:
            if close_sockets:
                self._parent_sock.close()
                self._child_sock.close()  # parent process also has reference to the child socket

        if transmit_error:
            raise RuntimeError("Failed to transmit the shared memory handles.")

    def recv_shareable_handles(self, close_sockets: bool = True):
        """Receive the handles of the shared memory slots over a local connection.
        This method should be called only from the child process.
        It needs to be called after the object is unpickled.
        Otherwise the child process will not have access to the shared memory"""
        transmit_error = True
        try:
            for slot in self._queue:
                slot.receive_shareable_handles(self._child_sock)
            transmit_error = False
        finally:
            if close_sockets:
                self._child_sock.close()

        if transmit_error:
            raise RuntimeError("Failed to receive the shared memory handles.")

    def put_nowait(self, item: dict[str, torch.Tensor], item_id: int) -> None:
        """Put item in the queue immediately. Should raise an exception if the queue is full."""
        with self._current_size.get_lock(), self._front_ind.get_lock():
            if self._current_size.value == self.max_size:
                raise Full

            # Calculate the location at which to insert the new item
            ind = (self._front_ind.value + self._current_size.value) % self.max_size
            self._current_size.value = self._current_size.value + 1

            self._position_locks[ind].acquire()

        self._queue[ind].write(item, item_id, True)
        self._position_locks[ind].release()

    def get_nowait(self) -> tuple[int, torch.Tensor]:
        """Get the oldest item from the queue immediately. Should raise an exception if the queue is empty."""
        if self._output_slot is None or self.output_device == "cpu":
            self._output_slot = self._queue[0].get_non_shared_empty_memory_slot(self.output_device)

        with self._current_size.get_lock():
            if self._current_size.value == 0:
                raise Empty

            with self._front_ind.get_lock():
                ind = self._front_ind.value
                self._current_size.value = self._current_size.value - 1
                self._front_ind.value = (self._front_ind.value + 1) % self.max_size

                self._position_locks[ind].acquire()

        item_id, item = self._queue[ind].read(self._output_slot, self.reading_synchronised)
        self._position_locks[ind].release()

        return item_id, item


# consumer
def worker(queue: CUDAQueue):
    import time

    _ = create_logger()
    checkCudaErrors(cuda.cuInit(0))
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, queue.device))
    queue.recv_shareable_handles()
    queue.logger.info(f"Is original process: {queue.is_original_process}")
    start_time = time.perf_counter()
    while time.perf_counter() - start_time < 5:
        item = queue.get()
        if item is not None:
            queue.logger.info(f"Received {item[0]}")
    checkCudaErrors(cuda.cuCtxDestroy(cuContext))


if __name__ == "__main__":
    import time

    mp.set_start_method("spawn")
    dtype = torch.float32
    _ = create_logger()

    # producer
    checkCudaErrors(cuda.cuInit(0))
    cu_device = checkCudaErrors(cuda.cuDeviceGet(0))
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cu_device))
    queue = CUDAQueue(10, dtype, "cuda", "cuda", {"a": (1000, 1000), "b": (1, 100000)}, add_batch_dim=True)

    data = [
        {
            "a": torch.rand((1000, 1000), dtype=dtype, device="cuda"),
            "b": torch.rand((1, 100000), dtype=dtype, device="cuda"),
        }
        for _ in range(100)
    ]

    p = mp.Process(target=worker, args=(queue,))
    p.start()
    queue.send_shareable_handles(p.pid)
    time.sleep(2)
    print("WARMUP")
    for i in range(3):
        queue.logger.info("Sending start")
        queue.put(data[i], i)
        queue.logger.info(f"Sent: {i}")
        time.sleep(0.5)
    time.sleep(2)
    print("TEST")
    for i in range(100):
        queue.logger.info("Sending start")
        queue.put(data[i], i)
        queue.logger.info(f"Sent: {i}")
        time.sleep(0.001)

    p.join()
    queue.close()
    checkCudaErrors(cuda.cuCtxDestroy(cuContext))
