import logging
from logging import Logger

# -------------
# Logging: logging + stdio = ✅, print + stdio = ❌ 
# -------------
def get_logger(name: str, log_file: str = "multitools-server.log", level: str = logging.DEBUG) -> Logger:
    """Single logger instance in module level

    Args:
        name: Specific name of the logger
        log_file: File name where log will be print (default: mcp server name)
        level: Level of logging
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        fmt = "%(asctime)s -- %(levelname)s -- %(name)s -- %(message)s"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(file_handler)
    return logger

