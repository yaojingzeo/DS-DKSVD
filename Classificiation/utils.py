import logging
import os


def setup_logger(log_dir, log_filename, log_level=logging.INFO):
    # 确保日志文件夹存在
    os.makedirs(log_dir, exist_ok=True)

    # 创建 logger
    logger = logging.getLogger(log_filename)
    logger.setLevel(log_level)

    # 创建文件处理器
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setLevel(log_level)

    # 创建控制台处理器
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger