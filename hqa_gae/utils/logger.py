import time
import pytz
import os
import logging
from datetime import datetime, timedelta
from pytorch_lightning.loggers import TensorBoardLogger

def shanghai_converter(sec):
    if time.strftime('%z') == "+0800":
        return datetime.now().timetuple()
    return (datetime.now() + timedelta(hours=8)).timetuple()

def get_time():
    tz = pytz.timezone('Asia/Shanghai')  
    now = datetime.now()
    now = now.astimezone(tz)
    time_str = now.strftime("%Y-%m-%d_%H:%M:%S")
    return time_str

class Logger(TensorBoardLogger):
    def __init__(self, save_dir, name, version, **kwargs):
        super().__init__(save_dir, name, version, **kwargs)
        py_logger = logging.getLogger(save_dir)
        if not py_logger.handlers:
            # 避免重复添加 handler 造成重复输出
            handler1 = logging.StreamHandler()
            handler2 = logging.FileHandler(filename=os.path.join(save_dir, name, version, "run.log"), encoding="utf-8")

            py_logger.setLevel(logging.DEBUG)
            handler1.setLevel(logging.WARNING)
            handler2.setLevel(logging.DEBUG)
            formatter1 = logging.Formatter("%(message)s")
            formatter2 = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            formatter2.converter = shanghai_converter
            handler1.setFormatter(formatter1)
            handler2.setFormatter(formatter2)
            py_logger.addHandler(handler1)
            py_logger.addHandler(handler2)

        self.py_logger = py_logger

    def log_info(self, msg):
        self.py_logger.info(msg)

    def log_warning(self, msg):
        self.py_logger.warning(msg)

    def log_error(self, msg):
        self.py_logger.error(msg)


class PyLogger(object):
    
    @staticmethod
    def get_logger(log_dir):
        logger = logging.getLogger(log_dir)
        if not logger.handlers:
            # 避免重复添加 handler 造成重复输出
            handler1 = logging.StreamHandler()
            handler2 = logging.FileHandler(filename=os.path.join(log_dir, "run.log"), encoding="utf-8")
            # 接受所有级别的 log
            logger.setLevel(logging.DEBUG)
            handler1.setLevel(logging.DEBUG)
            handler2.setLevel(logging.DEBUG)
            formatter1 = logging.Formatter("%(message)s")
            formatter2 = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            formatter2.converter = shanghai_converter
            handler1.setFormatter(formatter1)
            handler2.setFormatter(formatter2)
            logger.addHandler(handler1)
            logger.addHandler(handler2)

        return logger