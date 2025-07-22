import pynvml
import torch
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='gpu_memory_log.txt',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


# 初始化 pynvml
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 假设使用 GPU 0
    logging.info("pynvml initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize pynvml: {e}")
    handle = None

def log_gpu_memory_stats(event_description: str):
    """
    记录并打印当前GPU显存状态。
    :param event_description: 描述当前发生的事件，例如 "LLM_Inference_Start"
    """
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available.")
        return

    # 使用 pynvml 获取全局显存信息 (类似 nvidia-smi)
    total_mem = "N/A"
    used_mem = "N/A"
    free_mem = "N/A"
    if handle:
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mem = mem_info.total / 1024**2
        used_mem = mem_info.used / 1024**2
        free_mem = mem_info.free / 1024**2

    # 使用 PyTorch API 获取 PyTorch 内部的显存信息
    # torch.cuda.memory_allocated(): 当前被 PyTorch tensor 占用的显存
    # torch.cuda.memory_reserved(): PyTorch 缓存分配器预留的总显存
    allocated_mem = torch.cuda.memory_allocated(0) / 1024**2
    reserved_mem = torch.cuda.memory_reserved(0) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(0) / 1024**2

    log_message = (
        f"Event: [{event_description}] | "
        f"pynvml_Used: {used_mem:.2f} MB | "
        f"Torch_Allocated: {allocated_mem:.2f} MB | "
        f"Torch_Reserved: {reserved_mem:.2f} MB | "
        f"Torch_Max_Reserved_Peak: {max_reserved:.2f} MB"
    )
    logging.info(log_message)

def cleanup():
    """在程序结束时关闭pynvml"""
    if handle:
        pynvml.nvmlShutdown()
        logging.info("pynvml shut down.")
