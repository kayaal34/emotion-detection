import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dataset_prepare import EMOTIONS, TRAIN_ROWS, pixels_to_image, prepare, split_for


def pixels(value):
    return " ".join([str(value)] * 48 * 48)


def test_pixels_to_image():
    image = pixels_to_image(pixels(7))

    assert image.size == (48, 48)
    assert np.asarray(image).max() == 7


def test_pixels_to_image_rejects_wrong_size():
    with pytest.raises(ValueError):
        pixels_to_image("1 2 3")


def test_split_uses_usage_column():
    assert split_for(0, "Training") == "train"
    assert split_for(0, "PublicTest") == "test"
    assert split_for(0, "PrivateTest") == "test"


def test_split_falls_back_to_row_number():
    assert split_for(TRAIN_ROWS - 1) == "train"
    assert split_for(TRAIN_ROWS) == "test"


def test_prepare_saves_images_in_folders(tmp_path):
    df = pd.DataFrame({
        "emotion": [0, 3, 3, 6],
        "pixels": [pixels(1), pixels(2), pixels(3), pixels(4)],
        "Usage": ["Training", "Training", "PublicTest", "Training"],
    })

    counts = prepare(df, str(tmp_path))

    assert counts == {("train", "angry"): 1, ("train", "happy"): 1, ("test", "happy"): 1, ("train", "neutral"): 1}
    assert sorted(os.listdir(tmp_path / "train")) == sorted(EMOTIONS)
    saved = Image.open(tmp_path / "test" / "happy" / "im0.png")
    assert np.asarray(saved).max() == 3
