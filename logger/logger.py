import logging
import sys

def setup_logger(log_file='gpu_memory_log.txt', log_level=logging.INFO):
    """
    setup logger
    :param log_file: 日志文件路径
    :param log_level: 日志级别
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='w'
    )
    
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    return logging.getLogger('')
