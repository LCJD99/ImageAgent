"""Text translation model using T5."""
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import time
from typing import Dict, Any
from logger import log_gpu_memory_stats


class TranslationModel:
    def __init__(self):
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting to load translation model weights...")
        log_gpu_memory_stats("Translation_Model_Loading_Start")
        
        self.model_name = "t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished loading translation model weights")
        log_gpu_memory_stats("Translation_Model_Loading_Finish")
        
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "fr") -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (default: 'en' for English)
            target_lang: Target language code (default: 'fr' for French)
            
        Returns:
            Translated text
        """
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting translation from {source_lang} to {target_lang}")
        log_gpu_memory_stats("Translation_Prediction_Start")
        
        # Format input for T5
        task_prefix = f"translate {source_lang} to {target_lang}: "
        input_text = task_prefix + text
        
        # Tokenize the input
        input_ids = self.tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to(self.device)
        
        # Generate translation
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )
        
        # Decode the generated output
        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished translation")
        log_gpu_memory_stats("Translation_Prediction_Finish")
        
        # Swap model weights to CPU and clear GPU memory
        self._swap_to_cpu_and_clear_gpu()
        
        return translated_text
        
    def _swap_to_cpu_and_clear_gpu(self):
        """Swap model weights to CPU and clear GPU memory"""
        if self.device.type != "cuda":
            print("Model is already on CPU, no need to swap")
            return
            
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting to swap model weights to CPU")
        swap_start_time = time.time()
        log_gpu_memory_stats("Translation_Swap_To_CPU_Start")
        
        # Move model to CPU
        self.model.to("cpu")
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        swap_end_time = time.time()
        swap_duration = swap_end_time - swap_start_time
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished swapping model weights to CPU (took {swap_duration:.3f}s)")
        log_gpu_memory_stats("Translation_Swap_To_CPU_Finish")


_model = None

def get_translation_model():
    global _model
    if _model is None:
        _model = TranslationModel()
    return _model

def translate_text(text: str, source_lang: str = "en", target_lang: str = "fr") -> Dict[str, Any]:
    """
    Translate text from source language to target language.
    
    Args:
        text: Text to translate
        source_lang: Source language code (default: 'en' for English)
        target_lang: Target language code (default: 'fr' for French)
        
    Returns:
        Dictionary containing the translated text and metadata
    """
    model = get_translation_model()
    
    start_time = time.time()
    translated_text = model.translate(text, source_lang, target_lang)
    processing_time = time.time() - start_time
    
    return {
        "source_text": text,
        "source_language": source_lang,
        "target_language": target_lang,
        "translated_text": translated_text,
        "processing_time_seconds": round(processing_time, 3)
    }
