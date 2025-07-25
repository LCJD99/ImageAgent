"""AI Models package for image processing and language tasks."""

import logging
from typing import Optional

from .image_captioning import get_image_caption as _get_image_caption
from .object_detection import detect_objects_in_image as _detect_objects_in_image
from .image_classification import classify_image as _classify_image
from .translation import translate_text as _translate_text

# Default logger to use if none is provided
_default_logger = logging.getLogger("models")

def get_image_caption(image_path, logger=None):
    return _get_image_caption(image_path, logger=logger or _default_logger)

def detect_objects_in_image(image_path, threshold=None, logger=None):
    return _detect_objects_in_image(image_path, threshold=threshold, logger=logger or _default_logger)

def classify_image(image_path, top_k=None, logger=None):
    return _classify_image(image_path, top_k=top_k, logger=logger or _default_logger)

def translate_text(text, target_language=None, logger=None):
    return _translate_text(text, target_language=target_language, logger=logger or _default_logger)

__all__ = [
    "get_image_caption",
    "detect_objects_in_image",
    "classify_image",
    "translate_text"
]
