"""
示例：如何使用重构后的 logger 包进行 GPU 内存监控
"""
import torch
import atexit

# 方法1：使用重构后的 vmem_logger 包装器
from vmem_logger_new import log_gpu_memory_stats, cleanup

# 在程序退出时确保关闭 pynvml
atexit.register(cleanup)

# 记录初始状态
log_gpu_memory_stats("Initial State")

# 示例：分配一些 GPU 内存
if torch.cuda.is_available():
    # 创建一个大的 tensor
    tensor = torch.zeros((1000, 1000, 10), device='cuda')
    log_gpu_memory_stats("After tensor allocation")
    
    # 释放内存
    del tensor
    torch.cuda.empty_cache()
    log_gpu_memory_stats("After memory cleanup")


# 方法2：直接使用 logger 包 (更推荐的方式)
"""
import torch
import atexit
from logger import setup_logger, init_pynvml, log_gpu_memory_stats, cleanup

# 设置日志
setup_logger()

# 初始化 pynvml
init_pynvml()

# 在程序退出时确保关闭 pynvml
atexit.register(cleanup)

# 记录初始状态
log_gpu_memory_stats("Initial State")

# 示例：分配一些 GPU 内存
if torch.cuda.is_available():
    # 创建一个大的 tensor
    tensor = torch.zeros((1000, 1000, 10), device='cuda')
    log_gpu_memory_stats("After tensor allocation")
    
    # 释放内存
    del tensor
    torch.cuda.empty_cache()
    log_gpu_memory_stats("After memory cleanup")
"""
