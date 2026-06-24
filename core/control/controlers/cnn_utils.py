# cnn_utils.py
import cv2
import numpy as np



class Cnn_Utils:


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
        Convertit une image OpenCV en entrée CNN TFLite.
        input_shape attendu: [1, height, width, channels]
        """
        height = int(input_shape[1])
        width = int(input_shape[2])

        frame = cv2.resize(frame, (width, height))

        # Si image BGR OpenCV → RGB
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)

        return frame