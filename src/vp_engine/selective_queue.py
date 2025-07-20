from multiprocessing.managers import BaseManager
from time import time

from utils.logging_utils import create_logger, MESSAGE


class SelectiveQueue:
    def __init__(self, log_config: dict, maxsize: int = 10):
        self.queue = []
        self.maxsize = maxsize
        self.logger = create_logger(**log_config)

    def put(self, item):
        if len(self.queue) == self.maxsize:
            rmvd_item = self.queue.pop(0)
            self.logger.info(MESSAGE.MAX_QUEUE_SIZE_EXCEEDED.format(n=rmvd_item[1][0]))

        self.queue.append((time(), item))
        self.logger.info(MESSAGE.ADDED_TO_QUEUE.format(n=item[0], queue_size=len(self.queue)))

    def clear(self):
        self.queue = []

    def get(self, cutoff: float = float("-inf")):
        if len(self.queue) == 0:
            return None

        for i in range(len(self.queue)):
            if self.queue[i][0] > cutoff:
                return self.queue[i]

        return None


class SelectiveQueueManager(BaseManager):
    pass


SelectiveQueueManager.register("SelectiveQueue", SelectiveQueue)
