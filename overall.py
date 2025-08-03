import psutil
import time
import csv
from pynvml import *
import datetime

def get_cpu_memory_usage():
    """获取 CPU 使用的内存量 (MB)"""
    # psutil.virtual_memory() 返回系统内存使用情况
    # .used 是已使用的物理内存字节数
    # 除以 (1024**2) 转换为 MB
    return psutil.virtual_memory().used / (1024**2)

def get_gpu_memory_usage():
    """获取 NVIDIA GPU 使用的显存量 (MB)"""
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        gpu_mem_info = []
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            used_memory_mb = memory_info.used / (1024**2) # 已使用显存 (MB)
            gpu_mem_info.append({
                "index": i,
                "memory_used_mb": used_memory_mb
            })
        return gpu_mem_info
    except NVMLError as error:
        # print(f"NVIDIA NVML Error: {error}") # 在频繁写入时可以不打印此错误
        return None
    finally:
        try:
            nvmlShutdown()
        except NVMLError as error:
            # print(f"Error during NVML shutdown: {error}") # 同样，可以不打印
            pass

def main(output_filename="memory_monitor.csv"):
    """主函数，实时输出 CPU 和 GPU 内存使用情况并写入 CSV 文件"""
    print(f"--- 实时系统内存监控 (数据将写入 {output_filename}，按 Ctrl+C 停止) ---")

    headers = ["Timestamp", "CPU_Memory_Used_MB"]
    # 动态添加 GPU 相关的表头
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        for i in range(device_count):
            headers.append(f"GPU_{i}_Memory_Used_MB")
    except NVMLError:
        print("警告: 无法初始化 NVML，GPU 内存数据将不会写入。")
    finally:
        try:
            nvmlShutdown()
        except NVMLError:
            pass

    with open(output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(headers) # 写入表头

        while True:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] # 精确到毫秒
            row_data = [timestamp]

            # 获取 CPU 内存使用量
            cpu_mem_used = get_cpu_memory_usage()
            row_data.append(f"{cpu_mem_used:.2f}")

            # 获取 GPU 内存使用量
            gpu_mem_data = get_gpu_memory_usage()
            if gpu_mem_data:
                for gpu in gpu_mem_data:
                    row_data.append(f"{gpu['memory_used_mb']:.2f}")
            else:
                # 如果没有 GPU 数据，为每个预期 GPU 字段添加空值
                for _ in range(device_count): # 根据初始化时检测到的 GPU 数量填充空值
                    row_data.append("")
                # print("  未能获取 GPU 内存信息，请确保安装了 NVIDIA 驱动和 pynvml 库。") # 同样，可以不打印

            csv_writer.writerow(row_data)
            # print(f"写入数据: {row_data[0]} CPU 内存: {row_data[1]}MB ...") # 可以取消注释用于调试

            time.sleep(0.1) # 每0.1秒更新一次

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n--- 监控已停止，内存数据已保存到 CSV 文件 ---")
    except Exception as e:
        print(f"发生错误: {e}")
