import logging
import os
import sys
from loguru import logger as logurulogger
from config import config

LOG_FORMAT = (
    "<level>{level: <8}</level> "
    "{process.name} | "  # 进程名
    "{thread.name}  | "
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> - "
    "<blue>{process}</blue> "
    "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
LOG_NAME = ["uvicorn", "uvicorn.access", "uvicorn.error", "flask"]


class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logurulogger.level(record.levelname).name
        except AttributeError:
            level = logging._levelToName[record.levelno]

        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logurulogger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

class Logging:
    """自定义日志"""

    def __init__(self):
        self.log_path = "logs"
        os.makedirs(self.log_path, exist_ok=True)
        self._initlogger()
        self._reset_log_handler()
    

    def _initlogger(self):
        """初始化loguru配置"""
        logurulogger.remove()
        logurulogger.add(
            os.path.join(self.log_path, "error.log.{time:YYYY-MM-DD}"),
            format=LOG_FORMAT,
            level=logging.ERROR,
            rotation="00:00",
            retention="1 week",
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        logurulogger.add(
            os.path.join(self.log_path, "info.log.{time:YYYY-MM-DD}"),
            format=LOG_FORMAT,
            level=logging.INFO,
            rotation="00:00",
            retention="1 week",
            enqueue=True
        )
        logurulogger.add(
            sys.stdout,
            format=LOG_FORMAT,
            level=logging.DEBUG,
            colorize=True,
        )

        self.logger = logurulogger
        

    def _reset_log_handler(self):
        for log in LOG_NAME:
            logger = logging.getLogger(log)
            logger.handlers = [InterceptHandler()]

    def getlogger(self):
        return self.logger 

logger = Logging().getlogger()

