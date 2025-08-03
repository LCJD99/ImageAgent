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
from .colorizers import *

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import torch
import torch.nn.functional as F
from IPython import embed

class ColorizationModel:
    def __init__(self):
        # print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting to load colorization model weights...")
        log_gpu_memory_stats("Colorization_Model_Loading_Start")

        self.model = eccv16(pretrained=True).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        #print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished loading colorization model weights")
        log_gpu_memory_stats("Colorization_Model_Loading_Finish")

    def predict(self, image_path: str, output_path: str = None) -> str:
        """Colorize a grayscale image.

        Args:
            image_path: Path to the grayscale image file
            output_path: Optional path to save the colorized output

        Returns:
            Path to the colorized image or base64 encoded image if output_path is None
        """
        #print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting colorization prediction for {image_path}")
        log_gpu_memory_stats("Colorization_Prediction_Start")

        # Open and process the image
        img = load_img(image_path)
        (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
        tens_l_rs = tens_l_rs.cuda()

        output_image = postprocess_tens(tens_l_orig, self.model(tens_l_rs).cpu())

        result = 'output.jpg'
        plt.imsave(result, output_image)

        log_gpu_memory_stats("Colorization_Prediction_Finish")

        # Swap model weights to CPU and clear GPU memory
        #self._swap_to_cpu_and_clear_gpu()

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

def load_img(img_path):
	out_np = np.asarray(Image.open(img_path))
	if(out_np.ndim==2):
		out_np = np.tile(out_np[:,:,None],3)
	return out_np

def resize_img(img, HW=(256,256), resample=3):
	return np.asarray(Image.fromarray(img).resize((HW[1],HW[0]), resample=resample))

def preprocess_img(img_rgb_orig, HW=(256,256), resample=3):
	# return original size L and resized L as torch Tensors
	img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)

	img_lab_orig = color.rgb2lab(img_rgb_orig)
	img_lab_rs = color.rgb2lab(img_rgb_rs)

	img_l_orig = img_lab_orig[:,:,0]
	img_l_rs = img_lab_rs[:,:,0]

	tens_orig_l = torch.Tensor(img_l_orig)[None,None,:,:]
	tens_rs_l = torch.Tensor(img_l_rs)[None,None,:,:]

	return (tens_orig_l, tens_rs_l)

def postprocess_tens(tens_orig_l, out_ab, mode='bilinear'):
	# tens_orig_l 	1 x 1 x H_orig x W_orig
	# out_ab 		1 x 2 x H x W

	HW_orig = tens_orig_l.shape[2:]
	HW = out_ab.shape[2:]

	# call resize function if needed
	if(HW_orig[0]!=HW[0] or HW_orig[1]!=HW[1]):
		out_ab_orig = F.interpolate(out_ab, size=HW_orig, mode='bilinear')
	else:
		out_ab_orig = out_ab

	out_lab_orig = torch.cat((tens_orig_l, out_ab_orig), dim=1)
	return color.lab2rgb(out_lab_orig.data.cpu().numpy()[0,...].transpose((1,2,0)))
