"""Image classification model using Vision Transformer (ViT)."""
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import json
import time
from typing import List, Dict, Any
from logger.logger import setup_logger
from logger.gpu_memory import log_gpu_memory_stats

# Setup logger
logger = setup_logger(log_file='frontend_log.txt')

class ImageClassificationModel:
    def __init__(self):
        logger.info("Starting to load image classification model weights")
        start_time = time.time()
        log_gpu_memory_stats("ImageClassificationModel_LoadStart")

        self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        load_time = time.time() - start_time
        logger.info(f"Image classification model weights loaded in {load_time:.3f}s")
        log_gpu_memory_stats("ImageClassificationModel_LoadEnd")

    def classify_image(self, image_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Classify an image and return top predictions.

        Args:
            image_path: Path to the image file
            top_k: Number of top predictions to return

        Returns:
            List of top predictions with their probabilities
        """
        image = Image.open(image_path)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # Get top k predictions
        top_k_probs, top_k_indices = torch.topk(probabilities, top_k, dim=-1)

        predictions = []
        for prob, idx in zip(top_k_probs[0], top_k_indices[0]):
            prediction = {
                "class": self.model.config.id2label[idx.item()],
                "confidence": round(prob.item(), 4)
            }
            predictions.append(prediction)

        return predictions


# Global instance
_model_instance = None


def classify_image(image_path: str, top_k: str = "5") -> str:
    """Classify an image and return top predictions as JSON.

    Args:
        image_path: Path to the image file
        top_k: Number of top predictions to return (as string)

    Returns:
        JSON string containing classification results
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ImageClassificationModel()

    top_k_int = int(top_k)
    predictions = _model_instance.classify_image(image_path, top_k_int)

    result = {
        "image_path": image_path,
        "top_predictions": predictions,
        "predicted_class": predictions[0]["class"] if predictions else None
    }

    return json.dumps(result, indent=2)
