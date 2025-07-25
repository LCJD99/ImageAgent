"""Text translation model using T5."""
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import time
from typing import Dict, Any
from logger import log_gpu_memory_stats
from models.base_model import BaseModel


class TranslationModel(BaseModel):
    def __init__(self, weight_mode: str = "reserve"):
        super().__init__(weight_mode=weight_mode)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Starting to load translation model weights...")
        log_gpu_memory_stats("Translation_Model_Loading_Start")
        
        self.model_name = "t5-base"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        
        # Register model with the base class
        self.register_model("translation", self.model)
        
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}] Finished loading translation model weights")
        log_gpu_memory_stats("Translation_Model_Loading_Finish")
        
    def inference(self, text: str, source_lang: str = "en", target_lang: str = "fr") -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (default: 'en' for English)
            target_lang: Target language code (default: 'fr' for French)
            
        Returns:
            Translated text
        """
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
        
        return translated_text
        
    def translate(self, text: str, source_lang: str = "en", target_lang: str = "fr") -> str:
        """
        Public method that calls the inference method with proper weight management.
        """
        return self(text, source_lang, target_lang)


from models.weight_manager import get_global_weight_mode

_model = None

def get_translation_model():
    global _model
    if _model is None:
        _model = TranslationModel(weight_mode=get_global_weight_mode())
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
