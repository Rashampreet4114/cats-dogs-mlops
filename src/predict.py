"""Shared inference utilities — used by training (final eval), the FastAPI
service, and unit tests, so there is exactly one code path for "how the model
turns an image into a prediction"."""

import io
from pathlib import Path

import torch
from PIL import Image

from src.data_processing import CLASSES, preprocess_image
from src.model import SimpleCNN


def load_model(model_path: str, device: str = "cpu") -> SimpleCNN:
    model = SimpleCNN()
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def predict_image(model: SimpleCNN, image: Image.Image, device: str = "cpu") -> dict:
    """Run inference on a single PIL image.

    Returns: {"label": "cat"|"dog", "probabilities": {"cat": float, "dog": float}}
    """
    tensor = preprocess_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor)
        prob_dog = torch.sigmoid(logit).item()
    prob_cat = 1.0 - prob_dog
    label = CLASSES[1] if prob_dog >= 0.5 else CLASSES[0]
    return {
        "label": label,
        "probabilities": {"cat": round(prob_cat, 4), "dog": round(prob_dog, 4)},
    }


def predict_bytes(model: SimpleCNN, image_bytes: bytes, device: str = "cpu") -> dict:
    image = Image.open(io.BytesIO(image_bytes))
    return predict_image(model, image, device=device)


def predict_path(model: SimpleCNN, image_path: str, device: str = "cpu") -> dict:
    image = Image.open(Path(image_path))
    return predict_image(model, image, device=device)
