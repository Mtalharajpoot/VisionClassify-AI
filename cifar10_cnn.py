"""CIFAR-10 CNN training and inference pipeline.

Architecture follows the supplied step-by-step guide:
Input -> Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool
-> Conv2D(64) -> Flatten -> Dense(64) -> Dense(10, softmax)
"""

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "cifar10_cnn.keras"
OUTPUT_DIR = ROOT / "outputs"


def load_cifar10():
    """Load the CIFAR-10 train/test sets."""
    return datasets.cifar10.load_data()


def normalize_images(train_images: np.ndarray, test_images: np.ndarray):
    """Convert uint8 pixel values from 0-255 to float32 values from 0-1."""
    train = train_images.astype("float32") / 255.0
    test = test_images.astype("float32") / 255.0
    return train, test


def build_model() -> tf.keras.Model:
    """Build the CNN specified in the project guide."""
    model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="cifar10_cnn",
    )
    return model


def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    """Compile using Adam, sparse categorical crossentropy and accuracy."""
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def plot_training_history(history: tf.keras.callbacks.History) -> None:
    """Save training/validation accuracy and loss plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_curve.png")
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png")
    plt.close()


def train_model(
    model: tf.keras.Model,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 64,
):
    """Train the CNN using a 20% validation split."""
    return model.fit(
        train_images,
        train_labels,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
    )


def evaluate_model(
    model: tf.keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
):
    """Evaluate the trained model on the independent test set."""
    return model.evaluate(test_images, test_labels, verbose=1)


def predict_image(model: tf.keras.Model, image: np.ndarray) -> int:
    """Return the predicted class index for one preprocessed image."""
    prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
    return int(np.argmax(prediction[0]))


def save_model(model: tf.keras.Model, path: Path = MODEL_PATH) -> None:
    """Save the trained Keras model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def main() -> None:
    (train_images, train_labels), (test_images, test_labels) = load_cifar10()

    print("Training images:", train_images.shape)
    print("Training labels:", train_labels.shape)
    print("Test images:", test_images.shape)
    print("Test labels:", test_labels.shape)

    train_images, test_images = normalize_images(train_images, test_images)

    model = build_model()
    model = compile_model(model)
    model.summary()

    history = train_model(
        model,
        train_images,
        train_labels,
        epochs=10,
        batch_size=64,
    )

    plot_training_history(history)

    test_loss, test_accuracy = evaluate_model(
        model, test_images, test_labels
    )
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    index = 10
    predicted_class = predict_image(model, test_images[index])
    actual_class = int(test_labels[index][0])

    print("Actual:", CLASS_NAMES[actual_class])
    print("Predicted:", CLASS_NAMES[predicted_class])

    save_model(model)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
