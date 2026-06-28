#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
import tensorflow as tf
import numpy as np

from cnn_model import ZumiCNN


IMG_HEIGHT = 120
IMG_WIDTH = 160
IMG_CHANNELS = 3


def build_keras_cnn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),

        tf.keras.layers.Conv2D(16, kernel_size=5, strides=2, activation="relu"),
        tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, activation="relu"),
        tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, activation="relu"),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(100, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(2, activation="tanh"),
    ])

    return model


def convert():
    checkpoint_path = Path("best_cnn.pt")
    export_dir = Path("cnn_export")
    export_dir.mkdir(exist_ok=True)

    tflite_path = export_dir / "zumi_cnn.tflite"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Modèle introuvable: {checkpoint_path}")

    # Charger modèle PyTorch
    torch_model = ZumiCNN()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    torch_model.load_state_dict(checkpoint["model_state_dict"])
    torch_model.eval()

    # Créer modèle Keras équivalent
    keras_model = build_keras_cnn()

    print("⚠️ Attention : ce script crée l’architecture TFLite CNN.")
    print("Pour une conversion fidèle des poids PyTorch → Keras, il faut copier les poids couche par couche.")
    print("Version recommandée pour votre PFE : entraîner directement le CNN en TensorFlow/Keras OU exporter via ONNX.")

    # Sauvegarder modèle Keras non entraîné pour tester le pipeline TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"Modèle TFLite créé: {tflite_path}")


if __name__ == "__main__":
    convert()