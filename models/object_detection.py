"""Object detection model using DETR."""
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import json
import time
import logging
from typing import List, Dict, Any, Optional
from logger.gpu_memory import log_gpu_memory_stats

class ObjectDetectionModel:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Starting to load object detection model weights")
        start_time = time.time()
        log_gpu_memory_stats("ObjectDetectionModel_LoadStart")

        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-101", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-101", revision="no_timm")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        load_time = time.time() - start_time
        self.logger.info(f"Object detection model weights loaded in {load_time:.3f}s")
        log_gpu_memory_stats("ObjectDetectionModel_LoadEnd")

    def detect_objects(self, image_path: str, threshold: float = 0.9) -> List[Dict[str, Any]]:
        """Detect objects in an image.

        Args:
            image_path: Path to the image file
            threshold: Confidence threshold for detections

        Returns:
            List of detected objects with their information
        """
        image = Image.open(image_path)

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Convert outputs to COCO API format
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            detection = {
                "label": self.model.config.id2label[label.item()],
                "confidence": round(score.item(), 3),
                "bounding_box": box
            }
            detections.append(detection)

        return detections


# Global instance
_model_instance = None


def detect_objects_in_image(image_path: str, threshold: str = "0.9", logger: Optional[logging.Logger] = None) -> str:
    """Detect objects in an image and return results as JSON.

    Args:
        image_path: Path to the image file
        threshold: Confidence threshold for detections (as string)
        logger: Logger instance to use. If None, a default logger will be used.

    Returns:
        JSON string containing detected objects
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ObjectDetectionModel(logger=logger)

    threshold_float = float(threshold) if threshold else 0.9
    detections = _model_instance.detect_objects(image_path, threshold_float)

    result = {
        "image_path": image_path,
        "threshold": threshold_float,
        "detections": detections,
        "total_objects": len(detections)
    }

    return json.dumps(result, indent=2)
