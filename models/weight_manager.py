"""Model weight management utilities."""
import logging
from typing import Dict, Any

# Global weight mode setting for all models
_WEIGHT_MODE = "reserve"  # Default to keep models in GPU memory

def set_global_weight_mode(mode: str):
    """
    Set the global weight management mode for all models.
    
    Args:
        mode: Either "swap" or "reserve"
            - "swap": Move model weights to CPU after use, freeing GPU memory
            - "reserve": Keep model weights on GPU for faster subsequent calls
            
    Returns:
        bool: True if mode was valid and set, False otherwise
    """
    global _WEIGHT_MODE
    if mode.lower() in ["swap", "reserve"]:
        _WEIGHT_MODE = mode.lower()
        logging.info(f"Set global weight mode to: {_WEIGHT_MODE}")
        return True
    else:
        logging.warning(f"Invalid weight mode: {mode}. Must be 'swap' or 'reserve'")
        return False

def get_global_weight_mode() -> str:
    """
    Get the current global weight management mode.
    
    Returns:
        str: Current weight mode ("swap" or "reserve")
    """
    return _WEIGHT_MODE
