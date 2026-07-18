# cnn_utils.py


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def extract_frame_from_state(state):
    return state.frame

    # def extract_frame_from_state(state):
    #     """
    #     Essaie de récupérer l'image caméra depuis SensorState.
    #     Adapte les noms si votre projet utilise un autre attribut.
    #     """
    #     for attr in ["frame", "camera_frame", "image", "latest_frame", "raw_frame"]:
    #         if hasattr(state, attr):
    #             frame = getattr(state, attr)
    #             if frame is not None:
    #                 return frame
    #     return None

def preprocess_frame(frame, input_shape):
    """
    Prépare une frame BGR OpenCV pour un CNN TFLite NCHW.

    Entrée TFLite :
        [1, 3, height, width]
    """
    if frame is None:
        raise ValueError("Frame caméra absente")

    channels = int(input_shape[1])
    height = int(input_shape[2])
    width = int(input_shape[3])

    if channels != 3:
        raise ValueError(
            f"Nombre de canaux inattendu : {channels}"
        )

    frame = cv2.resize(
        frame,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    frame = frame.astype(np.float32) / 255.0

    # HWC → CHW
    frame = np.transpose(
        frame,
        (2, 0, 1)
    )

    # CHW → NCHW
    frame = np.expand_dims(
        frame,
        axis=0
    )

    return np.ascontiguousarray(
        frame,
        dtype=np.float32
    )