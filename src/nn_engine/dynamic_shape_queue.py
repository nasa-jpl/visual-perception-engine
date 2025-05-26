import torch
import multiprocessing as mp

from nn_engine.cuda_queue import FIFOQueueInterface


class DynamicShapeQueue(FIFOQueueInterface):
    def __init__(self, **kwargs):
        if "max_size" in kwargs:
            kwargs["maxsize"] = kwargs.pop("max_size")
        self.queue = mp.Queue(**kwargs)
        super().__init__()

    def put_nowait(self, obj: dict[str, torch.Tensor], item_id: int) -> None:
        """Immediately put item in the queue. Should raise an exception if the queue is full.
        IPC is not available on Jetson so we need to make sure that the tensor is on CPU before pickling.
        WARNING: pickling is slow and should be avoided if possible."""
        for key, value in obj.items():
            obj[key] = value.cpu()
        return self.queue.put_nowait((item_id, obj))

    def get_nowait(self):
        return self.queue.get_nowait()

    def put(self, obj: dict[str, torch.Tensor], item_id: int) -> None:
        """IPC is not available on Jetson so we need to make sure that the tensor is on CPU before pickling.
        WARNING: pickling is slow and should be avoided if possible."""
        for key, value in obj.items():
            obj[key] = value.cpu()
        return super().put((item_id, obj))

    def close(self):
        self.queue.close()
        self.queue.join_thread()
