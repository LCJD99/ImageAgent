import psutil
import logging
import time
import threading
import os
from datetime import datetime

# Get current process object for monitoring
current_process = psutil.Process(os.getpid())

def log_cpu_stats(event_description: str):
    """
    Record and print current process's CPU and memory usage stats.
    :param event_description: Description of the current event, e.g. "LLM_Inference_Start"
    """
    # Get CPU usage of current process (percentage of one CPU core)
    try:
        proc_cpu_percent = current_process.cpu_percent(interval=0.1)
        
        # Get memory info of current process
        proc_memory_info = current_process.memory_info()
        proc_mem_used = proc_memory_info.rss / (1024**2)  # RSS (Resident Set Size) in MB
        proc_mem_virtual = proc_memory_info.vms / (1024**2)  # VMS (Virtual Memory Size) in MB
        
        # Get number of threads used by this process
        proc_num_threads = current_process.num_threads()
        
        # Get CPU time accumulated by this process
        cpu_times = current_process.cpu_times()
        user_time = cpu_times.user
        system_time = cpu_times.system
        
        # Get system memory info for reference
        sys_memory = psutil.virtual_memory()
        sys_mem_total = sys_memory.total / (1024**2)
        
        log_message = (
            f"Event: [{event_description}] | "
            f"Process_CPU: {proc_cpu_percent:.2f}% | "
            f"Process_Memory_RSS: {proc_mem_used:.2f} MB | "
            f"Process_Memory_Virtual: {proc_mem_virtual:.2f} MB | "
            f"Threads: {proc_num_threads} | "
            f"CPU_Time: {user_time:.2f}s user, {system_time:.2f}s system | "
            f"System_Total_Memory: {sys_mem_total:.2f} MB"
        )
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
        log_message = f"Event: [{event_description}] | Error monitoring process: {str(e)}"
    
    logging.info(log_message)

def start_continuous_monitoring(interval=0.1, output_file='process_usage.csv', stop_event=None):
    """
    Start continuous monitoring of current process's CPU and memory usage in a separate thread
    
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
            f.write("timestamp,elapsed,proc_cpu_percent,proc_memory_rss_mb,proc_memory_vms_mb,proc_threads,proc_fds,proc_ctx_switches\n")
            
        start_time = time.time()
        logging.info(f"Starting continuous process monitoring (interval: {interval}s)")
        
        # Enable CPU monitoring for the process
        current_process.cpu_percent()  # First call will return 0.0, so we call once before entering the loop
        
        while not stop_event.is_set():
            try:
                # Get process CPU and memory information
                proc_cpu_percent = current_process.cpu_percent()
                proc_memory_info = current_process.memory_info()
                proc_rss_mb = proc_memory_info.rss / (1024**2)  # Resident Set Size in MB
                proc_vms_mb = proc_memory_info.vms / (1024**2)  # Virtual Memory Size in MB
                
                # Get additional process metrics
                proc_threads = current_process.num_threads()
                try:
                    proc_fds = len(current_process.open_files())  # Number of open file descriptors
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    proc_fds = -1  # If we can't access this info
                
                ctx_switches = current_process.num_ctx_switches()
                total_ctx_switches = ctx_switches.voluntary + ctx_switches.involuntary
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                elapsed = time.time() - start_time
                
                # Write to file
                with open(output_file, 'a') as f:
                    f.write(f"{timestamp},{elapsed:.3f},{proc_cpu_percent:.2f},{proc_rss_mb:.2f},{proc_vms_mb:.2f},{proc_threads},{proc_fds},{total_ctx_switches}\n")
                
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                logging.error(f"Error monitoring process: {str(e)}")
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        logging.info("Process monitoring stopped")
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitoring_worker, daemon=True)
    monitor_thread.start()
    
    return monitor_thread, stop_event

def stop_continuous_monitoring(monitor_thread, stop_event):
    """
    Stop the continuous process monitoring
    
    :param monitor_thread: The monitoring thread to stop
    :param stop_event: The event to signal to stop monitoring
    """
    if monitor_thread and monitor_thread.is_alive():
        stop_event.set()
        monitor_thread.join(timeout=2.0)
        logging.info("Process monitoring thread joined")
