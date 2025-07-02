import logging
import os
import sys
from loguru import logger as logurulogger
import json

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
        # self._connect_redis()
        os.makedirs(self.log_path, exist_ok=True)
        self._initlogger()
        self._reset_log_handler()
    
    def _connect_redis(self):
        retry = Retry(ExponentialBackoff(), 3)  # 重试3次，指数退避
        self.redis_client = redis.Redis(connection_pool=redis_pool,retry=retry)  # 使用连接池

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

        # logurulogger.add(self._log_to_redis, level="INFO", format=LOG_FORMAT)
        self.logger = logurulogger
        

    def _reset_log_handler(self):
        for log in LOG_NAME:
            logger = logging.getLogger(log)
            logger.handlers = [InterceptHandler()]

    def getlogger(self):
        return self.logger 

logger = Logging().getlogger()

