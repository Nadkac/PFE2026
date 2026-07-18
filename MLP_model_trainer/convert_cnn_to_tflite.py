#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversion du CNN Zumi PyTorch vers TensorFlow Lite.

Exemple PowerShell :

python .\convert_cnn_to_tflite.py `
    --model ".\cnn_checkpoints\best_cnn.pt" `
    --output ".\cnn_export\zumi_cnn.tflite"
"""

import ai_edge_torch
import litert_torch
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from cnn_model import ZumiCNN


DEFAULT_INPUT_SHAPE = [1, 3, 120, 160]
MOTOR_SPEED_MAX = 50.0


def load_pytorch_model(model_path: Path):
    """Charge le checkpoint et reconstruit le modèle CNN."""

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
            "Le checkpoint ne contient pas la clé "
            "'model_state_dict'."
        )

    model = ZumiCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_shape = checkpoint.get(
        "input_shape",
        DEFAULT_INPUT_SHAPE
    )

    input_shape = [int(value) for value in input_shape]

    print("[Chargement]")
    print(f"  Checkpoint : {model_path.resolve()}")
    print(f"  Entrée     : {input_shape}")
    print(f"  Sortie     : {checkpoint.get('output_dim', 2)}")

    val_loss = checkpoint.get("val_loss")
    if val_loss is not None:
        print(f"  Val loss   : {float(val_loss):.6f}")

    return model, checkpoint, input_shape


def convert_to_tflite(
    model: torch.nn.Module,
    input_shape: list[int],
    output_path: Path
):
    """Convertit directement le modèle PyTorch vers TFLite."""

    try:
        import ai_edge_torch
    except ImportError as exc:
        raise ImportError(
            "Le package ai-edge-torch n'est pas installé.\n"
            "Commande : python -m pip install ai-edge-torch"
        ) from exc

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    sample_input = torch.rand(
        *input_shape,
        dtype=torch.float32
    )

    print("\n[Conversion]")
    print(f"  Exemple d'entrée : {tuple(sample_input.shape)}")

    with torch.no_grad():
        torch_output = model(sample_input)

    print(f"  Sortie PyTorch   : {tuple(torch_output.shape)}")

    edge_model = ai_edge_torch.convert(
        model.eval(),
        (sample_input,)
    )

    edge_model.export(str(output_path))

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(f"  Modèle créé      : {output_path.resolve()}")
    print(f"  Taille           : {size_mb:.2f} Mo")

    return sample_input, torch_output


def export_metadata(
    checkpoint: dict,
    input_shape: list[int],
    output_dir: Path
):
    """Exporte les informations de prétraitement."""

    metadata = {
        "model_type": "cnn",
        "input_layout": "NCHW",
        "input_shape": input_shape,
        "image_height": input_shape[2],
        "image_width": input_shape[3],
        "channels": input_shape[1],
        "input_dtype": "float32",
        "pixel_normalization": "divide_by_255",
        "color_format_training": "RGB",
        "output_dim": int(checkpoint.get("output_dim", 2)),
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

    metadata_path = output_dir / "cnn_metadata.json"

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

    print(f"[Métadonnées] {metadata_path.resolve()}")

    return metadata_path


def verify_tflite_model(
    tflite_path: Path,
    model: torch.nn.Module,
    input_shape: list[int]
):
    """Compare une inférence PyTorch à une inférence TFLite."""

    try:
        import tensorflow as tf
    except ImportError:
        print(
            "\n[Vérification] TensorFlow absent. "
            "Vérification ignorée."
        )
        return

    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_path)
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("\n[Vérification TFLite]")
    print(
        f"  Entrée TFLite : "
        f"{input_details[0]['shape']} "
        f"{input_details[0]['dtype']}"
    )
    print(
        f"  Sortie TFLite : "
        f"{output_details[0]['shape']} "
        f"{output_details[0]['dtype']}"
    )

    test_input = np.random.rand(
        *input_shape
    ).astype(np.float32)

    with torch.no_grad():
        torch_output = model(
            torch.from_numpy(test_input)
        ).cpu().numpy()

    expected_shape = tuple(
        int(value)
        for value in input_details[0]["shape"]
    )

    if tuple(test_input.shape) != expected_shape:
        raise ValueError(
            "La forme d'entrée générée ne correspond pas "
            f"à celle du modèle TFLite : "
            f"{test_input.shape} contre {expected_shape}"
        )

    interpreter.set_tensor(
        input_details[0]["index"],
        test_input
    )
    interpreter.invoke()

    tflite_output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    max_difference = float(
        np.max(
            np.abs(torch_output - tflite_output)
        )
    )

    print(f"  PyTorch  : {torch_output.flatten()}")
    print(f"  TFLite   : {tflite_output.flatten()}")
    print(f"  Écart max: {max_difference:.8f}")

    if np.allclose(
        torch_output,
        tflite_output,
        atol=1e-4,
        rtol=1e-4
    ):
        print("  Résultat : conversion cohérente.")
    else:
        print(
            "  Avertissement : les résultats diffèrent "
            "au-delà de la tolérance."
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Conversion du CNN Zumi PyTorch vers TFLite"
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path("cnn_checkpoints") / "best_cnn.pt",
        help="Chemin du checkpoint PyTorch"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cnn_export") / "zumi_cnn.tflite",
        help="Chemin du fichier TFLite"
    )

    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Ne pas exécuter la vérification TFLite"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("CONVERSION CNN PYTORCH VERS TFLITE")
    print("=" * 65)

    try:
        model, checkpoint, input_shape = (
            load_pytorch_model(args.model)
        )

        convert_to_tflite(
            model=model,
            input_shape=input_shape,
            output_path=args.output
        )

        export_metadata(
            checkpoint=checkpoint,
            input_shape=input_shape,
            output_dir=args.output.parent
        )

        if not args.skip_verification:
            verify_tflite_model(
                tflite_path=args.output,
                model=model,
                input_shape=input_shape
            )

    except Exception as exc:
        print(f"\n[ERREUR] {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 65)
    print("CONVERSION TERMINÉE")
    print(f"Modèle   : {args.output.resolve()}")
    print(
        f"Metadata : "
        f"{(args.output.parent / 'cnn_metadata.json').resolve()}"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()