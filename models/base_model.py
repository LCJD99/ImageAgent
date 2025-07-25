"""Base model class with memory management options."""
import torch
from enum import Enum
import time
import logging
from typing import Optional, Dict, Any, Union, List
from logger import log_gpu_memory_stats

class ModelWeightMode(Enum):
    """Enum for model weight management modes."""
    SWAP = "swap"       # Move weights to CPU after use
    RESERVE = "reserve"  # Keep weights on GPU after use

class BaseModel:
    """
    Base model class with weight management options.
    Provides functionality to move model weights between GPU and CPU.
    """
    def __init__(self, weight_mode: str = "reserve"):
        """
        Initialize the base model with a weight mode.
        
        Args:
            weight_mode: Either "swap" or "reserve" (default: "reserve")
                - "swap": Move model weights to CPU after use, freeing GPU memory
                - "reserve": Keep model weights on GPU for faster subsequent calls
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set weight mode
        try:
            self.weight_mode = ModelWeightMode(weight_mode.lower())
        except ValueError:
            logging.warning(f"Invalid weight mode '{weight_mode}'. Defaulting to 'reserve'.")
            self.weight_mode = ModelWeightMode.RESERVE
            
        self.models: Dict[str, torch.nn.Module] = {}
        self.model_on_gpu = True
        
        logging.info(f"Model initialized with weight mode: {self.weight_mode.value}")
        
    def register_model(self, name: str, model: torch.nn.Module):
        """Register a PyTorch model for weight management."""
        self.models[name] = model
        # Move model to GPU initially
        if self.device.type == "cuda":
            model.to(self.device)
            self.model_on_gpu = True
            
    def to_gpu(self):
        """Move all registered models to GPU."""
        if not self.model_on_gpu and self.device.type == "cuda":
            start_time = time.time()
            log_gpu_memory_stats(f"{self.__class__.__name__}_MovingToGPU_Start")
            
            for name, model in self.models.items():
                model.to(self.device)
                
            self.model_on_gpu = True
            duration = time.time() - start_time
            logging.info(f"Moved models to GPU in {duration:.3f}s")
            log_gpu_memory_stats(f"{self.__class__.__name__}_MovingToGPU_End")
            
    def to_cpu(self):
        """Move all registered models to CPU, freeing GPU memory."""
        if self.model_on_gpu and self.weight_mode == ModelWeightMode.SWAP:
            start_time = time.time()
            log_gpu_memory_stats(f"{self.__class__.__name__}_MovingToCPU_Start")
            
            for name, model in self.models.items():
                model.to("cpu")
                
            # Force CUDA memory cleanup
            torch.cuda.empty_cache()
            
            self.model_on_gpu = False
            duration = time.time() - start_time
            logging.info(f"Moved models to CPU in {duration:.3f}s")
            log_gpu_memory_stats(f"{self.__class__.__name__}_MovingToCPU_End")
            
    def __call__(self, *args, **kwargs):
        """Execute model inference with automatic GPU/CPU management."""
        # Ensure model is on GPU before inference
        self.to_gpu()
        
        try:
            # Execute the actual inference
            result = self.inference(*args, **kwargs)
            
            # Handle post-inference weight management
            if self.weight_mode == ModelWeightMode.SWAP:
                self.to_cpu()
                
            return result
            
        except Exception as e:
            # Handle any errors during inference
            logging.error(f"Error during model inference: {str(e)}")
            
            # Try to free GPU memory if there was an error
            if self.weight_mode == ModelWeightMode.SWAP:
                self.to_cpu()
                
            raise
            
    def inference(self, *args, **kwargs):
        """
        Implement the model-specific inference logic in subclasses.
        
        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement the inference method")
