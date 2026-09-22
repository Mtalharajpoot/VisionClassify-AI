import numpy as np

from src.cifar10_cnn import CLASS_NAMES, build_model, normalize_images


def test_class_count():
    assert len(CLASS_NAMES) == 10


def test_normalization():
    train = np.array([[[[0, 127, 255]]]], dtype=np.uint8)
    test = train.copy()
    normalized_train, normalized_test = normalize_images(train, test)

    assert normalized_train.dtype == np.float32
    assert normalized_train.min() == 0.0
    assert normalized_train.max() == 1.0
    assert np.array_equal(normalized_train, normalized_test)


def test_model_output_shape_and_name():
    model = build_model()
    assert model.name == "cifar10_cnn"
    assert model.output_shape == (None, 10)
