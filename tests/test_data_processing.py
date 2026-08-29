from PIL import Image

from src.data_processing import IMG_SIZE, preprocess_image


def test_preprocess_image_shape_and_range():
    image = Image.new("RGB", (500, 333), color=(120, 200, 40))
    tensor = preprocess_image(image)

    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)
    # Normalized with ImageNet stats, so values should be roughly within [-3, 3]
    assert tensor.min().item() > -5
    assert tensor.max().item() < 5


def test_preprocess_image_handles_grayscale():
    image = Image.new("L", (100, 100), color=128)
    tensor = preprocess_image(image)

    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)
