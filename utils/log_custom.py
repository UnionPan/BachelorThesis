# -*- coding: utf-8 -*-

import logging
import logging.handlers
import os

# 日志文件的路径

# 日志文件的路径，FileHandler不能创建目录，这里先检查目录是否存在，不存在创建他
# 当然也可以继承之后重写FileHandler的构造函数
LOG_FILE_PATH = "test.log"
dir = os.path.dirname(LOG_FILE_PATH)
if not os.path.isdir(dir) and dir != '':
    os.mkdir(dir)
# 写入文件的日志等级，由于是详细信息，推荐设为debug
FILE_LOG_LEVEL = "DEBUG"
# 控制台的日照等级，info和warning都可以，可以按实际要求定制
CONSOLE_LOG_LEVEL = "INFO"
# 缓存日志等级，最好设为error或者critical
MEMOEY_LOG_LEVEL = "ERROR"
# 致命错误等级
URGENT_LOG_LEVEL = "CRITICAL"

MAPPING = {"CRITICAL": 50,
           "ERROR": 40,
           "WARNING": 30,
           "INFO": 20,
           "DEBUG": 10,
           "NOTSET": 0,
           }


class CustomLogger:
    """
    logger的配置
    """

    def __init__(self, logFile, file_level, console_level):
        self.logfile = logFile
        self.file_level = file_level
        self.console_level = console_level
        # 生成root logger
        self.logger = logging.getLogger("crawler")
        self.logger.setLevel(MAPPING[file_level])
        # 生成RotatingFileHandler
        self.fh = logging.handlers.RotatingFileHandler(logFile, mode='w', encoding="utf-8")
        self.fh.setLevel(MAPPING[file_level])
        # 生成StreamHandler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(MAPPING[console_level])
        # 设置格式
        formatter = logging.Formatter("%(asctime)s [%(levelname)-10s] : %(message)s", '%Y-%m-%d %H:%M:%S')
        self.ch.setFormatter(formatter)
        self.fh.setFormatter(formatter)
        # 把所有的handler添加到root logger中
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def debug(self, msg):
        if msg is not None:
            self.logger.debug(msg)

    def info(self, msg):
        if msg is not None:
            self.logger.info(msg)

    def flush(self):
        self.logger.removeHandler(self.fh)
        self.logger.removeHandler(self.ch)
        self.fh.close()
        self.ch.close()
        # 生成root logger
        # self.logger = logging.getLogger("crawler")
        # self.logger.setLevel(MAPPING[self.file_level])
        # 生成RotatingFileHandler
        self.fh = logging.handlers.RotatingFileHandler(self.logfile, mode='w', encoding="utf-8")
        self.fh.setLevel(MAPPING[self.file_level])
        # 生成StreamHandler
        self.ch = logging.StreamHandler()
        self.ch.setLevel(MAPPING[self.console_level])
        # 设置格式
        formatter = logging.Formatter("%(asctime)s [%(levelname)-10s] : %(message)s", '%Y-%m-%d %H:%M:%S')
        self.ch.setFormatter(formatter)
        self.fh.setFormatter(formatter)
        # 把所有的handler添加到root logger中
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def warning(self, msg):
        if msg is not None:
            self.logger.warning(msg)

    def error(self, msg):
        if msg is not None:
            self.logger.error(msg)

    def critical(self, msg):
        if msg is not None:
            self.logger.critical(msg)


log = CustomLogger(LOG_FILE_PATH, FILE_LOG_LEVEL, CONSOLE_LOG_LEVEL)


#def log_shutdown():
#    logging.shutdown()
#    log = CustomLogger(LOG_FILE_PATH, FILE_LOG_LEVEL, CONSOLE_LOG_LEVEL)
if __name__ == "__main__":
    # 测试代码
    for i in range(50):
        log.error(i)
        log.debug(i)
    log.critical("Database has gone away")
