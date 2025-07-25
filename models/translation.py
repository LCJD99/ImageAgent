"""Translation model using T5."""
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import time
from logger.logger import setup_logger
from logger.gpu_memory import log_gpu_memory_stats

# Setup logger
logger = setup_logger(log_file='frontend_log.txt')

class TranslationModel:
    def __init__(self):
        logger.info("Starting to load translation model weights")
        start_time = time.time()
        log_gpu_memory_stats("TranslationModel_LoadStart")

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        load_time = time.time() - start_time
        logger.info(f"Translation model weights loaded in {load_time:.3f}s")
        log_gpu_memory_stats("TranslationModel_LoadEnd")

    def translate(self, text: str, target_language: str = "en") -> str:
        """
        Translate text to the target language.

        Args:
            text: The text to translate
            target_language: Target language code (default: en)

        Returns:
            The translated text
        """
        # Prepend the task prefix for T5
        task_prefix = f"translate to {target_language}: "
        input_text = task_prefix + text

        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=512,
            num_beams=4,
            early_stopping=True
        )

        translated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

# Global instance
_model_instance = None

def translate_text(text: str, target_language: str = "en") -> str:
    """
    Translate text to the target language using the T5 model.

    Args:
        text: The text to translate
        target_language: Target language code (default: en)

    Returns:
        The translated text
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = TranslationModel()

    translated_text = _model_instance.translate(text, target_language)
    return translated_text
