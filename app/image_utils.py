import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(128, 128)):
    """
    Resize, normalize, and batchify the image for model input.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image.")

    image = image.convert("RGB")  # Ensure 3-channel input
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Optional: Validate shape
    assert image.shape == (1, target_size[0], target_size[1], 3), f"Unexpected shape: {image.shape}"

    return image
