import pynvml
import torch
import logging

handle = None

def init_pynvml():
    global handle
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        logging.debug("pynvml initialized successfully.")
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
    global handle
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

def start_continuous_monitoring(interval=0.1, output_file='vmem_usage.txt', stop_event=None):
    """
    Start continuous GPU memory monitoring in a separate thread

    :param interval: Time interval between measurements in seconds
    :param output_file: File to write the monitoring data
    :param stop_event: Threading event to signal when to stop monitoring
    :return: The monitoring thread object
    """
    import threading
    import time
    import os
    from datetime import datetime

    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Continuous monitoring disabled.")
        return None

    global handle
    if handle is None:
        init_pynvml()

    if stop_event is None:
        stop_event = threading.Event()

    def monitoring_worker():
        with open(output_file, 'w') as f:
            f.write("timestamp,event,used_memory_mb,allocated_memory_mb,reserved_memory_mb\n")

        start_time = time.time()
        # logging.info(f"Starting continuous GPU memory monitoring (interval: {interval}s)")

        while not stop_event.is_set():
            # Get memory information
            if handle:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mem = mem_info.used / 1024**2
            else:
                used_mem = 0

            allocated_mem = torch.cuda.memory_allocated(0) / 1024**2
            reserved_mem = torch.cuda.memory_reserved(0) / 1024**2

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            elapsed = time.time() - start_time

            # Write to file
            with open(output_file, 'a') as f:
                f.write(f"{timestamp},{elapsed:.3f},{used_mem:.2f},{allocated_mem:.2f},{reserved_mem:.2f}\n")

            # Sleep for the specified interval
            time.sleep(interval)

        # logging.info("GPU memory monitoring stopped")

    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitor_thread.start()

    return monitor_thread, stop_event

def stop_continuous_monitoring(monitor_thread, stop_event):
    """
    Stop the continuous GPU memory monitoring

    :param monitor_thread: The monitoring thread to stop
    :param stop_event: The event to signal to stop monitoring
    """
    if monitor_thread and monitor_thread.is_alive():
        stop_event.set()
        monitor_thread.join(timeout=2.0)
        # logging.info("GPU memory monitoring thread joined")

def cleanup():
    """在程序结束时关闭pynvml"""
    global handle
    if handle:
        pynvml.nvmlShutdown()
        logging.info("pynvml shut down.")
