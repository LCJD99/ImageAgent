"""Image captioning model using ViT-GPT2."""
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import json
import time
from typing import List
from logger.logger import setup_logger
from logger.gpu_memory import log_gpu_memory_stats

# Setup logger
logger = setup_logger(log_file='frontend_log.txt')


class ImageCaptioningModel:
    def __init__(self):
        logger.info("Starting to load image captioning model weights")
        start_time = time.time()
        log_gpu_memory_stats("ImageCaptioningModel_LoadStart")

        self.model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
        self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        load_time = time.time() - start_time
        logger.info(f"Image captioning model weights loaded in {load_time:.3f}s")
        log_gpu_memory_stats("ImageCaptioningModel_LoadEnd")

        self.max_length = 16
        self.num_beams = 1
        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

    def predict(self, image_paths: List[str]) -> List[str]:
        """Generate captions for given images.

        Args:
            image_paths: List of image file paths

        Returns:
            List of generated captions
        """
        images = []
        for image_path in image_paths:
            i_image = Image.open(image_path)
            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds


# Global instance
_model_instance = None


def get_image_caption(image_path: str) -> str:
    """Generate caption for a single image.

    Args:
        image_path: Path to the image file

    Returns:
        Generated caption as a string
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ImageCaptioningModel()

    captions = _model_instance.predict([image_path])
    return captions[0] if captions else "No caption generated"


def get_batch_image_captions(image_paths: str) -> str:
    """Generate captions for multiple images.

    Args:
        image_paths: Comma-separated list of image file paths

    Returns:
        JSON string containing captions for each image
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ImageCaptioningModel()

    paths_list = [path.strip() for path in image_paths.split(",")]
    captions = _model_instance.predict(paths_list)

    result = {}
    for path, caption in zip(paths_list, captions):
        result[path] = caption

    return json.dumps(result)
