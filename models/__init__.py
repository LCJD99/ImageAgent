"""AI Models package for image processing tasks."""

from .image_captioning import get_image_caption
from .object_detection import detect_objects_in_image
from .image_classification import classify_image

__all__ = [
    "get_image_caption",
    "detect_objects_in_image",
    "classify_image"
]
