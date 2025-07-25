import psutil
import logging
import time
import threading
from datetime import datetime

def log_cpu_stats(event_description: str):
    """
    Record and print current CPU usage stats.
    :param event_description: Description of the current event, e.g. "LLM_Inference_Start"
    """
    cpu_percent = psutil.cpu_percent(interval=0.1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    
    # Get per-core usage
    per_core = psutil.cpu_percent(interval=0.1, percpu=True)
    
    # Get memory usage
    memory = psutil.virtual_memory()
    mem_used = memory.used / (1024**2)
    mem_total = memory.total / (1024**2)
    mem_percent = memory.percent
    
    log_message = (
        f"Event: [{event_description}] | "
        f"CPU_Usage: {cpu_percent:.2f}% | "
        f"CPU_Cores: {cpu_count} | "
        f"CPU_Freq: {cpu_freq.current:.2f} MHz | "
        f"Memory_Used: {mem_used:.2f} MB ({mem_percent:.2f}%) of {mem_total:.2f} MB"
    )
    logging.info(log_message)

def start_continuous_monitoring(interval=0.1, output_file='cpu_usage.csv', stop_event=None):
    """
    Start continuous CPU monitoring in a separate thread
    
    :param interval: Time interval between measurements in seconds
    :param output_file: File to write the monitoring data
    :param stop_event: Threading event to signal when to stop monitoring
    :return: The monitoring thread object and stop event
    """
    if stop_event is None:
        stop_event = threading.Event()
    
    def monitoring_worker():
        # Initialize the CSV file with headers
        with open(output_file, 'w') as f:
            f.write("timestamp,elapsed,cpu_percent,memory_used_mb,memory_percent\n")
            
        start_time = time.time()
        logging.info(f"Starting continuous CPU monitoring (interval: {interval}s)")
        
        while not stop_event.is_set():
            # Get CPU and memory information
            cpu_percent = psutil.cpu_percent(interval=0.05)
            memory = psutil.virtual_memory()
            memory_used = memory.used / (1024**2)
            memory_percent = memory.percent
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            elapsed = time.time() - start_time
            
            # Write to file
            with open(output_file, 'a') as f:
                f.write(f"{timestamp},{elapsed:.3f},{cpu_percent:.2f},{memory_used:.2f},{memory_percent:.2f}\n")
            
            # Sleep for the specified interval (compensating for the time already spent in cpu_percent)
            time.sleep(max(0, interval - 0.05))
        
        logging.info("CPU monitoring stopped")
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitor_thread.start()
    
    return monitor_thread, stop_event

def stop_continuous_monitoring(monitor_thread, stop_event):
    """
    Stop the continuous CPU monitoring
    
    :param monitor_thread: The monitoring thread to stop
    :param stop_event: The event to signal to stop monitoring
    """
    if monitor_thread and monitor_thread.is_alive():
        stop_event.set()
        monitor_thread.join(timeout=2.0)
        logging.info("CPU monitoring thread joined")
