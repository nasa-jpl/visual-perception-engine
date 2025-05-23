from time import perf_counter

import torch
import torch.multiprocessing as mp


class Buffer:
    """A class that implements a buffer for storing data with timestamps. Based on a circular queue"""

    def __init__(
        self, buffer_size: int, data_signature: dict[str, tuple[int]], dtype: torch.dtype, add_batch_dim: bool = False
    ):
        self.buffer_size = buffer_size

        ### Allocate space and locks for data to be exchanged via this buffer ###
        self._locks = [mp.Lock() for _ in range(self.buffer_size)]  # Locks for each position in the buffer
        batch = [1] if add_batch_dim else []
        self._data = [
            {k: torch.empty(*batch, *v, dtype=dtype) for k, v in data_signature.items()}
            for _ in range(self.buffer_size)
        ]
        self._timestamps = mp.Array(
            "d", range(self.buffer_size), lock=False
        )  # Timestamps for each position in the buffer.

        self.newest_idx = mp.Value("i", self.buffer_size - 1)  # Index of the newest element
        self.oldest_idx = mp.Value("i", 0)  # Index of the oldest element

        for i in range(buffer_size):
            self._timestamps[i] = float("-inf")

            for k in data_signature.keys():
                self._data[i][k].share_memory_()

    def read_data(self, idx: int, last_time_stamp: float) -> tuple[float, dict[str, torch.Tensor]] | None:
        """Read the data at the given index, process safe"""
        with self._locks[idx]:  # ensure that when we read the data, no other process is writing it
            if self._timestamps[idx] == float("-inf") or self._timestamps[idx] <= last_time_stamp:
                return None
            out = self._timestamps[idx], {k: v.clone() for k, v in self._data[idx].items()}
            return out

    def write_data(self, idx, data) -> None:
        """Write the data at the given index, process safe"""
        with self._locks[idx]:  # ensure that when we write the data, no other process is reading it
            self._timestamps[idx] = perf_counter()
            for k, v in data.items():
                self._data[idx][k].copy_(v)

    def get(self, last_time_stamp: float) -> tuple[float, dict[str, torch.Tensor]] | None:
        """Return the newest data that is after the given timestamp"""
        newest_idx = self.newest_idx.value
        return self.read_data(newest_idx, last_time_stamp)

    def put(self, data: dict[str, torch.Tensor]) -> None:
        """Replaces the oldest element in the buffer with the given tensor"""
        with self.oldest_idx.get_lock():  # this lock is necessary to ensure that only one put is executed at a time
            self.write_data(self.oldest_idx.value, data)
            # after writing the data we need to update the newest and oldest indices
            self.newest_idx.value = self.oldest_idx.value
            self.oldest_idx.value = (self.oldest_idx.value + 1) % self.buffer_size

        # print(f"Newest: {self.newest_idx.value}, Oldest: {self.oldest_idx.value}")
        # print(f"Timestamps: {self._timestamps[:]}")
