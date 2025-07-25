"""AI Models package for image processing tasks and text translation."""

import torch
import time
from logger import log_gpu_memory_stats

from .image_captioning import get_image_caption
from .object_detection import detect_objects_in_image
from .image_classification import classify_image
from .colorization import colorize_image
from .translation import translate_text

def ensure_gpu_memory_cleared():
    """
    Helper function to ensure GPU memory is cleared properly.
    Call this function when you need to make sure GPU memory is freed up.
    """
    if torch.cuda.is_available():
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Clearing GPU memory...")
        log_gpu_memory_stats("GPU_Memory_Clear_Start")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        # Explicitly run garbage collection again
        gc.collect()
        
        log_gpu_memory_stats("GPU_Memory_Clear_End")
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] GPU memory cleared")

__all__ = [
    "get_image_caption",
    "detect_objects_in_image",
    "classify_image",
    "colorize_image",
    "translate_text",
    "ensure_gpu_memory_cleared"
]
