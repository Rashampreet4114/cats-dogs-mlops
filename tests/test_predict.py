from PIL import Image

from src.model import SimpleCNN
from src.predict import predict_image


def test_predict_image_returns_valid_result():
    model = SimpleCNN()  # untrained weights - only checking output contract
    model.eval()
    image = Image.new("RGB", (224, 224), color=(10, 10, 10))

    result = predict_image(model, image)

    assert result["label"] in ("cat", "dog")
    probs = result["probabilities"]
    assert set(probs.keys()) == {"cat", "dog"}
    assert abs(probs["cat"] + probs["dog"] - 1.0) < 1e-3
    assert 0.0 <= probs["cat"] <= 1.0
    assert 0.0 <= probs["dog"] <= 1.0
