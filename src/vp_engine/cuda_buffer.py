import os
import time
import socket
import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Literal
from contextlib import contextmanager

import torch
from cuda import cuda

from vp_engine.cuda_utils import CUDASharedMemorySlotWithIDAndTimestamp, checkCudaErrors
from utils.logging_utils import create_logger

@contextmanager
def nonblocking(lock):
    locked = lock.acquire(False)
    try:
        yield locked
    finally:
        if locked:
            lock.release()

class TimeBufferInterface(ABC):
    @abstractmethod
    def put(self, item) -> None:
        """Put item in the buffer, replace the oldest element if needed and return the respective timestamp."""
        pass

    @abstractmethod
    def get(self, last_timestamp) -> None | tuple[float, torch.Tensor]:
        """Get the most recent item with later timestamp than last_timestamp."""
        pass


class CUDATimeBuffer(TimeBufferInterface):
    """Buffer that will always output the most recent item that is newer than requested timestep.
    Adding new element to the buffer will overwrite the oldest element.
    Designed for one-to-many scenario, where one process writes to the buffer and many processes read from it.
    Although currently only one read at a time is allowed (via position locks), we cannot do better as we are using CUDA's memcopy, which is the bottleneck."""

    def __init__(
        self,
        max_size: int,
        dtype: torch.dtype,
        input_device: Literal["cpu", "cuda"],
        output_device: Literal["cpu", "cuda"],
        data_signature: dict[str, tuple[int]],
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
        self.device_id = device_id
        self.add_batch_dim = add_batch_dim
        self.logger = mp.get_logger()

        self._buffer = [
            CUDASharedMemorySlotWithIDAndTimestamp(
                data_signature, dtype, is_original_process, device_id=device_id, add_batch_dim=add_batch_dim
            )
            for _ in range(max_size)
        ]

        self._output_slot = (
            None  # memory slot where the output will be written, initialized the first time it is needed
        )

        self._position_locks = [mp.Lock() for _ in range(max_size)]
        self._newest_idx = mp.Value("i", max_size - 1)
        self._oldest_idx = mp.Value("i", 0)

        self._socket_lock = (
            mp.Lock()
        )  # to ensure that child processes receive shared memory handles only once and in order
        parent_sock, child_sock = socket.socketpair()
        os.set_inheritable(parent_sock.fileno(), True)
        os.set_inheritable(child_sock.fileno(), True)
        self._parent_sock = parent_sock
        self._child_sock = child_sock
        
        self.writeLock = LockedBufferWriter(self)

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
        return self._buffer[0].device

    def close(self):
        for slot in self._buffer:
            slot.close()

        # the sockets should be closed after sending and receiving the shared memory handles
        # if the handles are not transmitted, the sockets will be closed in the destructor
        self._child_sock.close()
        if self._parent_sock is not None:
            self._parent_sock.close()

    def send_shareable_handles(self, destination_pids: list[int], close_sockets: bool = True):
        """Send the handles of the shared memory slots over a local connection to all child processes.
        This method should be called only from the original process."""
        transmit_error = True
        try:
            for pid in destination_pids:
                for slot in self._buffer:
                    slot.send_shareable_handles(self._parent_sock, pid)
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
        Otherwise the child process will not have access to the shared memory."""
        transmit_error = True
        try:
            # There is multiple child processes, we need to ensure that each child process
            # receives the shared memory handles for all slots in the buffer only once and
            # that it happens in order.
            with self._socket_lock:
                for slot in self._buffer:
                    slot.receive_shareable_handles(self._child_sock)
            transmit_error = False
        finally:
            if close_sockets:
                self._child_sock.close()

        if transmit_error:
            raise RuntimeError("Failed to receive the shared memory handles.")

    def initialize_output_slot(self):
        """initialize persistent memory where the of the buffer will be written"""
        self._output_slot = {}
        for k, v in self._buffer[0].effective_data_signature.items():
            self._output_slot[k] = torch.empty(*v, dtype=self.dtype, device=self.device)

    def put(self, item: dict[str, torch.Tensor], item_id: int, sync=True) -> None:
        with self._oldest_idx.get_lock():
            oldest_idx = self._oldest_idx.value
            self._oldest_idx.value = (oldest_idx + 1) % self.max_size

        with self._position_locks[oldest_idx]:
            self._buffer[oldest_idx].write(item, item_id, sync)

        with self._newest_idx.get_lock():
            self._newest_idx.value = oldest_idx

    def get(self, last_timestamp: float, sync: bool = False) -> None | tuple[int, float, torch.Tensor]:
        # initialize the output slot the first time it is needed
        if self._output_slot is None or self.output_device == "cpu":
            self._output_slot = self._buffer[0].get_non_shared_empty_memory_slot(self.output_device)

        newest_idx = self._newest_idx.value
        with self._position_locks[newest_idx]:
            timestamp = self._buffer[newest_idx].get_timestamp()
            if timestamp == float("-inf") or timestamp <= last_timestamp:
                return None
            return self._buffer[newest_idx].read(self._output_slot, sync)
    
    def get_nowait(self, last_timestamp: float, sync: bool = False) -> None | tuple[int, float, torch.Tensor]:
        # initialize the output slot the first time it is needed
        if self._output_slot is None:
            self._output_slot = self._buffer[0].get_non_shared_empty_memory_slot(self.output_device)

        newest_idx = self._newest_idx.value
        with nonblocking(self._position_locks[newest_idx]) as locked:
            if locked:
                timestamp = self._buffer[newest_idx].get_timestamp()
                if timestamp == float("-inf") or timestamp <= last_timestamp:
                    return None
                return self._buffer[newest_idx].read(self._output_slot, sync)
            else:
                return None

class LockedBufferWriter:
    def __init__(self, buffer: CUDATimeBuffer):
        self.buffer = buffer
        self.current_lock = None
        self.oldest_idx = None        
        self.identifier = None
    
    def __call__(self, identifier):
            self.identifier = identifier
            return self
    
    def __enter__(self):
        with self.buffer._oldest_idx.get_lock():
            self.oldest_idx = self.buffer._oldest_idx.value
            self.buffer._oldest_idx.value = (self.oldest_idx + 1) % self.buffer.max_size
        
        self.current_lock = self.buffer._position_locks[self.oldest_idx]
        self.current_lock.acquire()
        
        return self.buffer._buffer[self.oldest_idx].memory_dict
            
    def __exit__(self, exc_type, exc_value, traceback):
        self.buffer._buffer[self.oldest_idx].timestamp.value = time.perf_counter()
        self.buffer._buffer[self.oldest_idx].id.value = self.identifier
        self.current_lock.release()
        with self.buffer._newest_idx.get_lock():
            self.buffer._newest_idx.value = self.oldest_idx


if __name__ == "__main__":
    # consumer
    def worker(buffer: CUDATimeBuffer, process_id: int):
        import time

        _ = create_logger()
        checkCudaErrors(cuda.cuInit(0))
        cuContext = checkCudaErrors(cuda.cuCtxCreate(0, buffer.device))
        buffer.recv_shareable_handles()
        start_time = time.perf_counter()
        last_timestamp = float("-inf")
        while time.perf_counter() - start_time < 10:
            item = buffer.get(last_timestamp)
            if item is not None:
                buffer.logger.info(f"Received: {item[0]}")
                last_timestamp = item[1]
        checkCudaErrors(cuda.cuCtxDestroy(cuContext))
        
    import time

    mp.set_start_method("spawn")
    dtype = torch.float32
    _ = create_logger()

    # producer
    checkCudaErrors(cuda.cuInit(0))
    cu_device = checkCudaErrors(cuda.cuDeviceGet(0))
    cuContext = checkCudaErrors(cuda.cuCtxCreate(0, cu_device))
    buffer = CUDATimeBuffer(
        10, dtype, "cuda", "cuda", {"a": (1000, 1000), "b": (1, 100000)}, reading_synchronised=True, add_batch_dim=True
    )

    data = [
        {
            "a": torch.rand((1000, 1000), dtype=dtype, device="cuda"),
            "b": torch.rand((1, 100000), dtype=dtype, device="cuda"),
        }
        for _ in range(100)
    ]

    processes = []
    for i in range(1, 5):
        p = mp.Process(target=worker, args=(buffer, i))
        p.start()
        processes.append(p)
    buffer.send_shareable_handles([p.pid for p in processes])
    time.sleep(2)
    print("WARMUP")
    for i in range(3):
        buffer.logger.info("Sending start")
        buffer.put(data[i], i)
        buffer.logger.info(f"Sent: {i}")
        time.sleep(0.5)
    time.sleep(2)
    print("TEST")
    for i in range(100):
        buffer.logger.info("Sending start")
        buffer.put(data[i], i)
        buffer.logger.info(f"Sent: {i}")
        time.sleep(0.001)

    for p in processes:
        p.join()
    buffer.close()
    checkCudaErrors(cuda.cuCtxDestroy(cuContext))
