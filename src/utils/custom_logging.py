from pathlib import Path
import logging


class CustomLogger():
    def __init__(self, name):
        # TODO: 設定外だし
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        handler_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_dir = Path('logs/')
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_dir/'train.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(handler_format)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger
