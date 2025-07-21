from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

import torch

image = Image.open('pic1.jpg')

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

