import logging
from rich.logging import RichHandler
from logging import FileHandler, getLogger

logger = getLogger("pipeline")

def setup_logger(name: str, level: str = "INFO", to_file: bool = True, log_file: str = "app.log"):
    """
    设置带有 rich 美化的日志
    :param name: logger 名称
    :param level: 日志级别，默认为 INFO
    :param to_file: 是否同时输出到文件，默认 True
    :param log_file: 日志文件名，默认为 app.log
    :return: 配置好的 logger
    """
    
    # 配置日志格式
    log_format = "%(asctime)s | %(levelname)s | %(message)s"
    
    # 创建 handlers
    handlers = [RichHandler(rich_tracebacks=True, show_time=False, show_level=False, markup=True)]  # 控制台输出
    
    # 如果需要输出到文件
    if to_file:
        file_handler = FileHandler(log_file, mode='w')
        handlers.append(file_handler)
    
    # 配置 logging
    logging.basicConfig(
        level=level,  # 设置日志级别
        format=log_format,  # 设置格式
        datefmt="[%X]",  # 时间格式
        handlers=handlers
    )
    
    logger = logging.getLogger(name)
    return logger
