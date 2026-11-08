import ast
import os

import numpy as np
import pytest

from dataset_prepare import EMOTIONS as FOLDERS

SRC = os.path.join(os.path.dirname(__file__), "..", "src", "emotions.py")


def emotion_labels():
    # Read the list without importing TensorFlow, so this runs in CI too
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    for node in tree.body:
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "EMOTIONS"):
            return ast.literal_eval(node.value)
    raise AssertionError("EMOTIONS not found")


def test_labels_match_the_alphabetical_folder_order():
    # flow_from_directory numbers the classes by sorted folder name
    assert [label.lower() for label in emotion_labels()] == sorted(FOLDERS)


def test_model_output_and_preprocessing():
    pytest.importorskip("tensorflow")
    pytest.importorskip("cv2")
    import emotions

    model = emotions.build_model()
    assert model.output_shape == (None, len(emotions.EMOTIONS))

    gray = np.full((200, 200), 255, dtype=np.uint8)
    faces = emotions.preprocess_faces(gray, [(10, 10, 100, 100), (50, 60, 80, 80)])
    assert faces.shape == (2, 48, 48, 1)
    assert faces.max() == pytest.approx(1.0)

    predictions = model.predict(faces, verbose=0)
    assert predictions.shape == (2, 7)
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)


def test_image_mode_needs_a_path():
    pytest.importorskip("tensorflow")
    pytest.importorskip("cv2")
    import emotions

    with pytest.raises(SystemExit):
        emotions.main(["--mode", "image"])
