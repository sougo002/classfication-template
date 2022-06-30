from time import time
import torch
import logging
from collections import OrderedDict

from utils.custom_logging import CustomLogger

logger = CustomLogger(__name__).get_logger()


class Timer:
    def __init__(self) -> None:
        self.time_dict = OrderedDict()

    def record(self, name: str):
        # カウントする
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record_time = time()
        self.time_dict[name.lower()] = record_time
        logger.debug(f'time recorded. name: {name}, time: {record_time}')

    def get_time_list(self):
        return self.time_dict

    def get_elapsed_time(self):
        return self.time_dict['end'] - self.time_dict['start']

    def get_time_between(self, from_name: str, to_name: str):
        return self.time_dict[to_name.lower()] - self.time_dict[from_name.lower()]