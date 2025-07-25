"""Image colorization model."""
from transformers import AutoModel
import torch
from PIL import Image
import numpy as np
import io
import base64
import os
import time
from logger import log_gpu_memory_stats


class ColorizationModel:
    def __init__(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting to load colorization model weights...")
        log_gpu_memory_stats("Colorization_Model_Loading_Start")
        
        self.model = AutoModel.from_pretrained("sebastiansarasti/AutoEncoderImageColorization")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished loading colorization model weights")
        log_gpu_memory_stats("Colorization_Model_Loading_Finish")

    def predict(self, image_path: str, output_path: str = None) -> str:
        """Colorize a grayscale image.
        
        Args:
            image_path: Path to the grayscale image file
            output_path: Optional path to save the colorized output
            
        Returns:
            Path to the colorized image or base64 encoded image if output_path is None
        """
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting colorization prediction for {image_path}")
        log_gpu_memory_stats("Colorization_Prediction_Start")
        
        # Open and process the image
        img = Image.open(image_path)
        
        # If image is already in RGB, convert to grayscale first to ensure proper colorization
        if img.mode == "RGB":
            # Convert to grayscale and back to RGB for processing
            img_gray = img.convert('L')
            img = img_gray.convert('RGB')
        elif img.mode != "RGB":
            img = img.convert('RGB')
        
        # Prepare input for the model
        input_tensor = torch.from_numpy(np.array(img)).float() / 255.0
        input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        # Generate colorized output
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
        
        # Convert output tensor to image
        output_image = output_tensor[0].cpu().permute(1, 2, 0).numpy()
        output_image = (output_image * 255).clip(0, 255).astype(np.uint8)
        output_image = Image.fromarray(output_image)
        
        # Save or return the colorized image
        if output_path:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            output_image.save(output_path)
            result = output_path
        else:
            # Return base64 encoded image
            buffer = io.BytesIO()
            output_image.save(buffer, format="JPEG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
            result = encoded_image
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished colorization prediction")
        log_gpu_memory_stats("Colorization_Prediction_Finish")
        
        # Swap model weights to CPU and clear GPU memory
        self._swap_to_cpu_and_clear_gpu()
        
        return result
        
    def _swap_to_cpu_and_clear_gpu(self):
        """Swap model weights to CPU and clear GPU memory"""
        if self.device.type != "cuda":
            print("Model is already on CPU, no need to swap")
            return
            
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting to swap model weights to CPU")
        swap_start_time = time.time()
        log_gpu_memory_stats("Colorization_Swap_To_CPU_Start")
        
        # Move model to CPU
        self.model.to("cpu")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        swap_end_time = time.time()
        swap_duration = swap_end_time - swap_start_time
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished swapping model weights to CPU (took {swap_duration:.3f}s)")
        log_gpu_memory_stats("Colorization_Swap_To_CPU_Finish")


# Global instance
_model_instance = None


def colorize_image(image_path: str, output_path: str = None) -> dict:
    """Colorize a single image.
    
    Args:
        image_path: Path to the image file
        output_path: Optional path to save the colorized output
        
    Returns:
        Dictionary containing the result information
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ColorizationModel()
    
    if not output_path:
        # Generate output filename by appending '_colorized' to the original filename
        base_dir = os.path.dirname(os.path.abspath(image_path))
        filename, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(base_dir, f"{filename}_colorized{ext}")
    
    result_path = _model_instance.predict(image_path, output_path)
    
    return {
        "original_image": image_path,
        "colorized_image": result_path,
        "status": "success"
    }
