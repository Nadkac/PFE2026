#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversion du CNN Zumi :

    PyTorch checkpoint
        -> modèle TensorFlow/Keras équivalent
        -> TensorFlow Lite

Commande :

    python3 convert_cnn_to_tflite.py

Avec chemins personnalisés :

    python3 convert_cnn_to_tflite.py \
        --model cnn_checkpoints/best_cnn.pt \
        --output cnn_export/zumi_cnn.tflite

Avec quantification :

    python3 convert_cnn_to_tflite.py --quantize
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    import tensorflow as tf
except ImportError as exc:
    raise ImportError(
        "TensorFlow n'est pas installé dans l'environnement actif.\n"
        "Vérifie avec : python3 -c \"import tensorflow as tf; "
        "print(tf.__version__)\""
    ) from exc

from cnn_model import ZumiCNN


DEFAULT_HEIGHT = 120
DEFAULT_WIDTH = 160
DEFAULT_CHANNELS = 3
DEFAULT_OUTPUT_DIM = 2
MOTOR_SPEED_MAX = 50.0


def load_pytorch_model(model_path: Path):
    """Charge le checkpoint PyTorch et reconstruit ZumiCNN."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"Checkpoint introuvable : {model_path.resolve()}"
        )

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Le checkpoint ne contient pas 'model_state_dict'."
        )

    model = ZumiCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_shape = checkpoint.get(
        "input_shape",
        [1, 3, DEFAULT_HEIGHT, DEFAULT_WIDTH]
    )

    input_shape = [int(value) for value in input_shape]

    if len(input_shape) != 4:
        raise ValueError(
            f"Forme d'entrée invalide : {input_shape}"
        )

    print("[Chargement PyTorch]")
    print(f"  Checkpoint : {model_path.resolve()}")
    print(f"  Entrée     : {input_shape} (NCHW)")
    print(f"  Sortie     : {checkpoint.get('output_dim', 2)}")

    val_loss = checkpoint.get("val_loss")
    if val_loss is not None:
        print(f"  Val loss   : {float(val_loss):.6f}")

    return model, checkpoint, input_shape


def create_tensorflow_model(
    image_height: int,
    image_width: int
) -> tf.keras.Model:
    """
    Crée l'équivalent TensorFlow du CNN PyTorch amélioré.

    PyTorch :
        Conv(3,24,5,stride=2,padding=2)
        BatchNorm + ReLU
        Conv(24,36,5,stride=2,padding=2)
        BatchNorm + ReLU
        Conv(36,48,5,stride=2,padding=2)
        BatchNorm + ReLU
        Conv(48,64,3,stride=1,padding=1)
        BatchNorm + ReLU
        Conv(64,64,3,stride=1,padding=1)
        BatchNorm + ReLU
        AdaptiveAvgPool(3,4)
        Dense 128
        Dense 64
        Dense 16
        Dense 2 + Tanh

    TensorFlow utilise NHWC :
        [batch, height, width, channels]
    """

    inputs = tf.keras.Input(
        shape=(image_height, image_width, 3),
        name="camera_image"
    )

    x = tf.keras.layers.Conv2D(
        filters=24,
        kernel_size=5,
        strides=2,
        padding="same",
        use_bias=True,
        name="conv1"
    )(inputs)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name="bn1"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu1"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=36,
        kernel_size=5,
        strides=2,
        padding="same",
        use_bias=True,
        name="conv2"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name="bn2"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu2"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=48,
        kernel_size=5,
        strides=2,
        padding="same",
        use_bias=True,
        name="conv3"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name="bn3"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu3"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=True,
        name="conv4"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name="bn4"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu4"
    )(x)

    x = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=True,
        name="conv5"
    )(x)

    x = tf.keras.layers.BatchNormalization(
        epsilon=1e-5,
        momentum=0.9,
        name="bn5"
    )(x)

    x = tf.keras.layers.ReLU(
        name="relu5"
    )(x)

    # Pour une entrée 120 x 160 :
    # après trois convolutions stride=2 : 15 x 20
    # AveragePooling 5 x 5 produit 3 x 4,
    # équivalent à AdaptiveAvgPool2d((3, 4)).
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(5, 5),
        strides=(5, 5),
        padding="valid",
        name="adaptive_average_pool"
    )(x)

    x = tf.keras.layers.Flatten(
        name="flatten"
    )(x)

    x = tf.keras.layers.Dense(
        units=128,
        activation="relu",
        name="dense1"
    )(x)

    # Dropout inutile pendant l'inférence.
    # Il n'est donc pas nécessaire dans le modèle TFLite.

    x = tf.keras.layers.Dense(
        units=64,
        activation="relu",
        name="dense2"
    )(x)

    x = tf.keras.layers.Dense(
        units=16,
        activation="relu",
        name="dense3"
    )(x)

    outputs = tf.keras.layers.Dense(
        units=2,
        activation="tanh",
        name="motor_output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="ZumiCNN"
    )

    # Initialise les variables Keras.
    dummy_input = np.zeros(
        (1, image_height, image_width, 3),
        dtype=np.float32
    )

    model(dummy_input, training=False)

    return model


def get_torch_layers(
    pytorch_model: nn.Module
):
    """Récupère les couches utiles dans leur ordre d'exécution."""

    conv_layers = [
        module
        for module in pytorch_model.features
        if isinstance(module, nn.Conv2d)
    ]

    batchnorm_layers = [
        module
        for module in pytorch_model.features
        if isinstance(module, nn.BatchNorm2d)
    ]

    linear_layers = [
        module
        for module in pytorch_model.regressor
        if isinstance(module, nn.Linear)
    ]

    return conv_layers, batchnorm_layers, linear_layers


def copy_conv2d_weights(
    torch_layer: nn.Conv2d,
    tf_layer: tf.keras.layers.Conv2D
):
    """
    Copie les poids Conv2d.

    PyTorch :
        [out_channels, in_channels, kernel_h, kernel_w]

    TensorFlow :
        [kernel_h, kernel_w, in_channels, out_channels]
    """

    torch_weight = (
        torch_layer.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    tf_weight = np.transpose(
        torch_weight,
        (2, 3, 1, 0)
    )

    if torch_layer.bias is not None:
        tf_bias = (
            torch_layer.bias
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    else:
        tf_bias = np.zeros(
            torch_layer.out_channels,
            dtype=np.float32
        )

    expected_kernel_shape = tuple(tf_layer.kernel.shape)

    if tf_weight.shape != expected_kernel_shape:
        raise ValueError(
            f"Incompatibilité Conv2D {tf_layer.name} : "
            f"poids convertis {tf_weight.shape}, "
            f"poids attendus {expected_kernel_shape}"
        )

    tf_layer.set_weights([
        tf_weight,
        tf_bias
    ])


def copy_batchnorm_weights(
    torch_layer: nn.BatchNorm2d,
    tf_layer: tf.keras.layers.BatchNormalization
):
    """
    Copie BatchNorm2d vers BatchNormalization.

    Ordre TensorFlow :
        gamma
        beta
        moving_mean
        moving_variance
    """

    gamma = (
        torch_layer.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    beta = (
        torch_layer.bias
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    moving_mean = (
        torch_layer.running_mean
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    moving_variance = (
        torch_layer.running_var
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    tf_layer.set_weights([
        gamma,
        beta,
        moving_mean,
        moving_variance
    ])


def copy_dense_weights(
    torch_layer: nn.Linear,
    tf_layer: tf.keras.layers.Dense
):
    """
    Copie une couche Linear vers Dense.

    PyTorch :
        [out_features, in_features]

    TensorFlow :
        [in_features, out_features]
    """

    torch_weight = (
        torch_layer.weight
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    tf_weight = torch_weight.T

    torch_bias = (
        torch_layer.bias
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    expected_kernel_shape = tuple(tf_layer.kernel.shape)

    if tf_weight.shape != expected_kernel_shape:
        raise ValueError(
            f"Incompatibilité Dense {tf_layer.name} : "
            f"poids convertis {tf_weight.shape}, "
            f"poids attendus {expected_kernel_shape}"
        )

    tf_layer.set_weights([
        tf_weight,
        torch_bias
    ])


def transfer_weights(
    pytorch_model: nn.Module,
    tensorflow_model: tf.keras.Model
):
    """Copie tous les poids PyTorch dans le modèle TensorFlow."""

    conv_layers, bn_layers, linear_layers = get_torch_layers(
        pytorch_model
    )

    tf_conv_names = [
        "conv1",
        "conv2",
        "conv3",
        "conv4",
        "conv5"
    ]

    tf_bn_names = [
        "bn1",
        "bn2",
        "bn3",
        "bn4",
        "bn5"
    ]

    tf_dense_names = [
        "dense1",
        "dense2",
        "dense3",
        "motor_output"
    ]

    print("\n[Transfert des poids]")
    print(f"  Conv2d PyTorch     : {len(conv_layers)}")
    print(f"  BatchNorm PyTorch  : {len(bn_layers)}")
    print(f"  Linear PyTorch     : {len(linear_layers)}")

    if len(conv_layers) != len(tf_conv_names):
        raise ValueError(
            "Le modèle PyTorch ne contient pas cinq couches Conv2d. "
            "Le fichier cnn_model.py ne correspond probablement pas "
            "à l'architecture attendue."
        )

    if len(bn_layers) != len(tf_bn_names):
        raise ValueError(
            "Le modèle PyTorch ne contient pas cinq couches BatchNorm2d."
        )

    if len(linear_layers) != len(tf_dense_names):
        raise ValueError(
            "Le modèle PyTorch ne contient pas quatre couches Linear."
        )

    for torch_layer, tf_name in zip(
        conv_layers,
        tf_conv_names
    ):
        tf_layer = tensorflow_model.get_layer(tf_name)

        copy_conv2d_weights(
            torch_layer,
            tf_layer
        )

        print(f"  OK : {tf_name}")

    for torch_layer, tf_name in zip(
        bn_layers,
        tf_bn_names
    ):
        tf_layer = tensorflow_model.get_layer(tf_name)

        copy_batchnorm_weights(
            torch_layer,
            tf_layer
        )

        print(f"  OK : {tf_name}")

    for torch_layer, tf_name in zip(
        linear_layers,
        tf_dense_names
    ):
        tf_layer = tensorflow_model.get_layer(tf_name)

        copy_dense_weights(
            torch_layer,
            tf_layer
        )

        print(f"  OK : {tf_name}")


def compare_pytorch_tensorflow(
    pytorch_model: nn.Module,
    tensorflow_model: tf.keras.Model,
    image_height: int,
    image_width: int
):
    """Compare les sorties PyTorch et TensorFlow."""

    np.random.seed(42)

    # Image normalisée comme lors de l'inférence.
    input_nhwc = np.random.rand(
        1,
        image_height,
        image_width,
        3
    ).astype(np.float32)

    # TensorFlow utilise NHWC.
    tensorflow_output = tensorflow_model(
        input_nhwc,
        training=False
    ).numpy()

    # PyTorch utilise NCHW.
    input_nchw = np.transpose(
        input_nhwc,
        (0, 3, 1, 2)
    )

    with torch.no_grad():
        pytorch_output = pytorch_model(
            torch.from_numpy(input_nchw)
        ).cpu().numpy()

    max_difference = float(
        np.max(
            np.abs(
                pytorch_output - tensorflow_output
            )
        )
    )

    print("\n[Comparaison PyTorch / TensorFlow]")
    print(f"  PyTorch   : {pytorch_output.flatten()}")
    print(f"  TensorFlow: {tensorflow_output.flatten()}")
    print(f"  Écart max : {max_difference:.8f}")

    if np.allclose(
        pytorch_output,
        tensorflow_output,
        atol=1e-4,
        rtol=1e-4
    ):
        print("  Résultat  : modèles cohérents")
    else:
        raise RuntimeError(
            "Les sorties PyTorch et TensorFlow ne sont pas "
            "suffisamment proches. La conversion est interrompue."
        )


def convert_to_tflite(
    tensorflow_model: tf.keras.Model,
    output_path: Path,
    quantize: bool = False
):
    """Convertit le modèle Keras vers TFLite."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(
        tensorflow_model
    )

    if quantize:
        # Quantification dynamique des poids.
        converter.optimizations = [
            tf.lite.Optimize.DEFAULT
        ]

        print("\n[TFLite] Quantification activée")

    tflite_model = converter.convert()

    with output_path.open("wb") as file:
        file.write(tflite_model)

    size_mb = (
        output_path.stat().st_size
        / (1024 * 1024)
    )

    print("\n[Conversion TFLite]")
    print(f"  Modèle : {output_path.resolve()}")
    print(f"  Taille : {size_mb:.2f} Mo")


def verify_tflite(
    tflite_path: Path,
    pytorch_model: nn.Module
):
    """Compare les sorties PyTorch et TFLite."""

    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path)
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n[Vérification TFLite]")
    print(
        f"  Entrée : "
        f"{input_details[0]['shape']} "
        f"{input_details[0]['dtype']}"
    )
    print(
        f"  Sortie : "
        f"{output_details[0]['shape']} "
        f"{output_details[0]['dtype']}"
    )

    expected_input_shape = (
        1,
        DEFAULT_HEIGHT,
        DEFAULT_WIDTH,
        DEFAULT_CHANNELS
    )

    actual_input_shape = tuple(
        int(value)
        for value in input_details[0]["shape"]
    )

    if actual_input_shape != expected_input_shape:
        raise ValueError(
            f"Forme TFLite inattendue : {actual_input_shape}. "
            f"Forme attendue : {expected_input_shape}"
        )

    np.random.seed(123)

    test_input_nhwc = np.random.rand(
        *expected_input_shape
    ).astype(np.float32)

    interpreter.set_tensor(
        input_details[0]["index"],
        test_input_nhwc
    )

    interpreter.invoke()

    tflite_output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    test_input_nchw = np.transpose(
        test_input_nhwc,
        (0, 3, 1, 2)
    )

    with torch.no_grad():
        pytorch_output = pytorch_model(
            torch.from_numpy(test_input_nchw)
        ).cpu().numpy()

    max_difference = float(
        np.max(
            np.abs(
                pytorch_output - tflite_output
            )
        )
    )

    print(f"  PyTorch : {pytorch_output.flatten()}")
    print(f"  TFLite  : {tflite_output.flatten()}")
    print(f"  Écart   : {max_difference:.8f}")

    if np.allclose(
        pytorch_output,
        tflite_output,
        atol=1e-3,
        rtol=1e-3
    ):
        print("  Résultat : conversion cohérente")
    else:
        print(
            "  AVERTISSEMENT : l'écart est supérieur "
            "à la tolérance de 0,001."
        )


def export_metadata(
    checkpoint: dict,
    output_directory: Path
):
    """Crée les métadonnées nécessaires sur le Raspberry Pi."""

    metadata = {
        "model_type": "cnn",
        "input_layout": "NHWC",
        "input_shape": [
            1,
            DEFAULT_HEIGHT,
            DEFAULT_WIDTH,
            DEFAULT_CHANNELS
        ],
        "image_height": DEFAULT_HEIGHT,
        "image_width": DEFAULT_WIDTH,
        "channels": DEFAULT_CHANNELS,
        "input_dtype": "float32",
        "pixel_normalization": "divide_by_255",
        "color_format": "RGB",
        "output_dim": int(
            checkpoint.get(
                "output_dim",
                DEFAULT_OUTPUT_DIM
            )
        ),
        "output_activation": "tanh",
        "output_range": [-1.0, 1.0],
        "motor_speed_max": float(
            checkpoint.get(
                "motor_speed_max",
                MOTOR_SPEED_MAX
            )
        ),
        "output_meaning": [
            "left_motor_normalized",
            "right_motor_normalized"
        ],
        "val_loss": checkpoint.get("val_loss")
    }

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata_path = (
        output_directory
        / "cnn_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\n[Métadonnées] {metadata_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Conversion manuelle du CNN PyTorch vers TFLite"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=(
            Path("cnn_checkpoints")
            / "best_cnn.pt"
        ),
        help="Chemin du checkpoint PyTorch"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path("cnn_export")
            / "zumi_cnn.tflite"
        ),
        help="Chemin du modèle TFLite"
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Active la quantification dynamique"
    )

    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Ignore la comparaison finale"
    )

    args = parser.parse_args()

    print("=" * 68)
    print("CONVERSION CNN PYTORCH -> TENSORFLOW -> TFLITE")
    print("=" * 68)

    try:
        pytorch_model, checkpoint, input_shape = (
            load_pytorch_model(args.model)
        )

        batch, channels, height, width = input_shape

        if batch != 1:
            raise ValueError(
                "Le batch du checkpoint doit être égal à 1."
            )

        if channels != 3:
            raise ValueError(
                "Le modèle doit accepter trois canaux RGB."
            )

        if height != DEFAULT_HEIGHT or width != DEFAULT_WIDTH:
            raise ValueError(
                f"Le script attend des images "
                f"{DEFAULT_HEIGHT} x {DEFAULT_WIDTH}, "
                f"mais le checkpoint indique "
                f"{height} x {width}."
            )

        tensorflow_model = create_tensorflow_model(
            image_height=height,
            image_width=width
        )

        tensorflow_model.summary()

        transfer_weights(
            pytorch_model,
            tensorflow_model
        )

        compare_pytorch_tensorflow(
            pytorch_model,
            tensorflow_model,
            image_height=height,
            image_width=width
        )

        convert_to_tflite(
            tensorflow_model,
            args.output,
            quantize=args.quantize
        )

        export_metadata(
            checkpoint,
            args.output.parent
        )

        if not args.skip_verification:
            verify_tflite(
                args.output,
                pytorch_model
            )

    except Exception as exc:
        print(f"\n[ERREUR] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 68)
    print("CONVERSION TERMINÉE")
    print(f"Modèle   : {args.output.resolve()}")
    print(
        "Metadata : "
        f"{(args.output.parent / 'cnn_metadata.json').resolve()}"
    )
    print("=" * 68)


if __name__ == "__main__":
    main()