import psutil
import logging
import time
import threading
import os
from datetime import datetime

"""
CPU and Physical Memory Monitoring Module

This module focuses on monitoring physical memory (RAM) usage rather than virtual memory.

Key Memory Concepts:
- Physical Memory (RAM): The actual hardware memory installed in the system
- Virtual Memory: A memory management technique that includes both RAM and swap space
- RSS (Resident Set Size): The portion of a process's memory that is held in physical RAM
- VMS (Virtual Memory Size): The total virtual memory used by a process

Note: psutil.virtual_memory() actually returns physical memory (RAM) information,
despite its confusing name. This is the industry standard naming convention.
"""

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
        proc_physical_mem_used = proc_memory_info.rss / (1024**2)  # RSS (Physical memory in RAM) in MB
        proc_virtual_mem_used = proc_memory_info.vms / (1024**2)  # VMS (Virtual Memory Size) in MB
        
        # Get number of threads used by this process
        proc_num_threads = current_process.num_threads()
        
        # Get CPU time accumulated by this process
        cpu_times = current_process.cpu_times()
        user_time = cpu_times.user
        system_time = cpu_times.system
        
        # Get system physical memory info (RAM)
        sys_physical_memory = psutil.virtual_memory()  # Note: virtual_memory() actually returns physical RAM info
        sys_physical_mem_total = sys_physical_memory.total / (1024**2)
        sys_physical_mem_used = sys_physical_memory.used / (1024**2)
        sys_physical_mem_available = sys_physical_memory.available / (1024**2)
        sys_physical_mem_percent = sys_physical_memory.percent
        
        # Get swap memory info
        swap_memory = psutil.swap_memory()
        swap_total = swap_memory.total / (1024**2)
        swap_used = swap_memory.used / (1024**2)
        swap_percent = swap_memory.percent
        
        log_message = (
            f"Event: [{event_description}] | "
            f"Process_CPU: {proc_cpu_percent:.2f}% | "
            f"Process_Physical_Memory: {proc_physical_mem_used:.2f} MB | "
            f"Process_Virtual_Memory: {proc_virtual_mem_used:.2f} MB | "
            f"Threads: {proc_num_threads} | "
            f"CPU_Time: {user_time:.2f}s user, {system_time:.2f}s system | "
            f"System_Physical_Memory: {sys_physical_mem_used:.2f}/{sys_physical_mem_total:.2f} MB ({sys_physical_mem_percent:.1f}%) | "
            f"Available_Physical_Memory: {sys_physical_mem_available:.2f} MB | "
            f"Swap_Memory: {swap_used:.2f}/{swap_total:.2f} MB ({swap_percent:.1f}%)"
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
            f.write("timestamp,elapsed,proc_cpu_percent,proc_physical_mem_mb,proc_virtual_mem_mb,proc_threads,proc_fds,proc_ctx_switches,sys_physical_mem_total_mb,sys_physical_mem_used_mb,sys_physical_mem_available_mb,sys_physical_mem_percent,swap_total_mb,swap_used_mb,swap_percent\n")
            
        start_time = time.time()
        logging.info(f"Starting continuous process monitoring (interval: {interval}s)")
        
        # Enable CPU monitoring for the process
        current_process.cpu_percent()  # First call will return 0.0, so we call once before entering the loop
        
        while not stop_event.is_set():
            try:
                # Get process CPU and memory information
                proc_cpu_percent = current_process.cpu_percent()
                proc_memory_info = current_process.memory_info()
                proc_physical_mem_mb = proc_memory_info.rss / (1024**2)  # Physical memory (RSS) in MB
                proc_virtual_mem_mb = proc_memory_info.vms / (1024**2)  # Virtual Memory Size in MB
                
                # Get additional process metrics
                proc_threads = current_process.num_threads()
                try:
                    proc_fds = len(current_process.open_files())  # Number of open file descriptors
                except (psutil.AccessDenied, psutil.ZombieProcess):
                    proc_fds = -1  # If we can't access this info
                
                ctx_switches = current_process.num_ctx_switches()
                total_ctx_switches = ctx_switches.voluntary + ctx_switches.involuntary
                
                # Get global system physical memory information
                sys_physical_memory = psutil.virtual_memory()  # Note: virtual_memory() returns physical RAM info
                sys_physical_mem_total_mb = sys_physical_memory.total / (1024**2)
                sys_physical_mem_used_mb = sys_physical_memory.used / (1024**2)
                sys_physical_mem_available_mb = sys_physical_memory.available / (1024**2)
                sys_physical_mem_percent = sys_physical_memory.percent
                
                # Get swap memory information
                swap_memory = psutil.swap_memory()
                swap_total_mb = swap_memory.total / (1024**2)
                swap_used_mb = swap_memory.used / (1024**2)
                swap_percent = swap_memory.percent
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                elapsed = time.time() - start_time
                
                # Write to file
                with open(output_file, 'a') as f:
                    f.write(f"{timestamp},{elapsed:.3f},{proc_cpu_percent:.2f},{proc_physical_mem_mb:.2f},{proc_virtual_mem_mb:.2f},{proc_threads},{proc_fds},{total_ctx_switches},{sys_physical_mem_total_mb:.2f},{sys_physical_mem_used_mb:.2f},{sys_physical_mem_available_mb:.2f},{sys_physical_mem_percent:.1f},{swap_total_mb:.2f},{swap_used_mb:.2f},{swap_percent:.1f}\n")
                
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

def log_global_memory_stats(event_description: str):
    """
    Record and print global system physical memory usage stats.
    :param event_description: Description of the current event, e.g. "Global_Physical_Memory_Check"
    """
    try:
        # Get global system physical memory information
        sys_physical_memory = psutil.virtual_memory()  # Note: virtual_memory() returns physical RAM info
        sys_physical_mem_total_gb = sys_physical_memory.total / (1024**3)  # Total physical memory in GB
        sys_physical_mem_used_gb = sys_physical_memory.used / (1024**3)    # Used physical memory in GB
        sys_physical_mem_available_gb = sys_physical_memory.available / (1024**3)  # Available physical memory in GB
        sys_physical_mem_percent = sys_physical_memory.percent
        sys_physical_mem_free_gb = sys_physical_memory.free / (1024**3)    # Free physical memory in GB
        sys_physical_mem_cached_gb = getattr(sys_physical_memory, 'cached', 0) / (1024**3)  # Cached physical memory in GB (Linux/macOS)
        sys_physical_mem_buffers_gb = getattr(sys_physical_memory, 'buffers', 0) / (1024**3)  # Buffer physical memory in GB (Linux)
        
        # Get swap memory information
        swap_memory = psutil.swap_memory()
        swap_total_gb = swap_memory.total / (1024**3)
        swap_used_gb = swap_memory.used / (1024**3)
        swap_free_gb = swap_memory.free / (1024**3)
        swap_percent = swap_memory.percent
        
        # Get number of active processes
        num_processes = len(psutil.pids())
        
        log_message = (
            f"Event: [{event_description}] | "
            f"System_Physical_Memory: {sys_physical_mem_used_gb:.2f}/{sys_physical_mem_total_gb:.2f} GB ({sys_physical_mem_percent:.1f}% used) | "
            f"Available: {sys_physical_mem_available_gb:.2f} GB | "
            f"Free: {sys_physical_mem_free_gb:.2f} GB | "
            f"Cached: {sys_physical_mem_cached_gb:.2f} GB | "
            f"Buffers: {sys_physical_mem_buffers_gb:.2f} GB | "
            f"Swap: {swap_used_gb:.2f}/{swap_total_gb:.2f} GB ({swap_percent:.1f}% used) | "
            f"Active_Processes: {num_processes}"
        )
        
    except Exception as e:
        log_message = f"Event: [{event_description}] | Error monitoring global memory: {str(e)}"
    
    logging.info(log_message)

def get_global_memory_info():
    """
    Get detailed global physical memory information as a dictionary.
    :return: Dictionary containing global physical memory metrics
    """
    try:
        # Get global system physical memory information
        sys_physical_memory = psutil.virtual_memory()  # Note: virtual_memory() returns physical RAM info
        swap_memory = psutil.swap_memory()
        
        return {
            'physical_memory': {
                'total_gb': sys_physical_memory.total / (1024**3),
                'used_gb': sys_physical_memory.used / (1024**3),
                'available_gb': sys_physical_memory.available / (1024**3),
                'free_gb': sys_physical_memory.free / (1024**3),
                'percent_used': sys_physical_memory.percent,
                'cached_gb': getattr(sys_physical_memory, 'cached', 0) / (1024**3),
                'buffers_gb': getattr(sys_physical_memory, 'buffers', 0) / (1024**3)
            },
            'swap_memory': {
                'total_gb': swap_memory.total / (1024**3),
                'used_gb': swap_memory.used / (1024**3),
                'free_gb': swap_memory.free / (1024**3),
                'percent_used': swap_memory.percent
            },
            'active_processes': len(psutil.pids())
        }
    except Exception as e:
        return {'error': str(e)}

def start_global_memory_monitoring(interval=1.0, output_file='global_physical_memory_usage.csv', stop_event=None):
    """
    Start continuous monitoring of global system physical memory usage in a separate thread
    
    :param interval: Time interval between measurements in seconds
    :param output_file: File to write the monitoring data
    :param stop_event: Threading event to signal when to stop monitoring
    :return: The monitoring thread object and stop event
    """
    if stop_event is None:
        stop_event = threading.Event()
    
    def global_memory_monitoring_worker():
        # Initialize the CSV file with headers
        with open(output_file, 'w') as f:
            f.write("timestamp,elapsed,sys_physical_mem_total_gb,sys_physical_mem_used_gb,sys_physical_mem_available_gb,sys_physical_mem_free_gb,sys_physical_mem_percent,sys_physical_mem_cached_gb,sys_physical_mem_buffers_gb,swap_total_gb,swap_used_gb,swap_free_gb,swap_percent,active_processes\n")
            
        start_time = time.time()
        logging.info(f"Starting continuous global physical memory monitoring (interval: {interval}s)")
        
        while not stop_event.is_set():
            try:
                # Get global system physical memory information
                sys_physical_memory = psutil.virtual_memory()  # Note: virtual_memory() returns physical RAM info
                sys_physical_mem_total_gb = sys_physical_memory.total / (1024**3)
                sys_physical_mem_used_gb = sys_physical_memory.used / (1024**3)
                sys_physical_mem_available_gb = sys_physical_memory.available / (1024**3)
                sys_physical_mem_free_gb = sys_physical_memory.free / (1024**3)
                sys_physical_mem_percent = sys_physical_memory.percent
                sys_physical_mem_cached_gb = getattr(sys_physical_memory, 'cached', 0) / (1024**3)
                sys_physical_mem_buffers_gb = getattr(sys_physical_memory, 'buffers', 0) / (1024**3)
                
                # Get swap memory information
                swap_memory = psutil.swap_memory()
                swap_total_gb = swap_memory.total / (1024**3)
                swap_used_gb = swap_memory.used / (1024**3)
                swap_free_gb = swap_memory.free / (1024**3)
                swap_percent = swap_memory.percent
                
                # Get number of active processes
                active_processes = len(psutil.pids())
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                elapsed = time.time() - start_time
                
                # Write to file
                with open(output_file, 'a') as f:
                    f.write(f"{timestamp},{elapsed:.3f},{sys_physical_mem_total_gb:.3f},{sys_physical_mem_used_gb:.3f},{sys_physical_mem_available_gb:.3f},{sys_physical_mem_free_gb:.3f},{sys_physical_mem_percent:.1f},{sys_physical_mem_cached_gb:.3f},{sys_physical_mem_buffers_gb:.3f},{swap_total_gb:.3f},{swap_used_gb:.3f},{swap_free_gb:.3f},{swap_percent:.1f},{active_processes}\n")
                
            except Exception as e:
                logging.error(f"Error monitoring global physical memory: {str(e)}")
            
            # Sleep for the specified interval
            time.sleep(interval)
        
        logging.info("Global physical memory monitoring stopped")
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=global_memory_monitoring_worker, daemon=True)
    monitor_thread.start()
    
    return monitor_thread, stop_event
