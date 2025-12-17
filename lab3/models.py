from tensorflow import keras
from tensorflow.keras import layers

def build_residual_denoiser(input_shape=(28, 28, 1), num_filters=32, num_layers=5, kernel_size=3):
    """
    Residual denoiser: predicts a correction to the noisy input.
    output = noisy + residual
    """
    inputs = keras.Input(shape=input_shape)

    x = inputs
    for _ in range(num_layers - 1):
        x = layers.Conv2D(num_filters, kernel_size, padding="same", activation="relu")(x)

    # Predict residual (same channels as input)
    residual = layers.Conv2D(input_shape[-1], kernel_size, padding="same", activation="linear")(x)

    outputs = layers.Add()([inputs, residual])

    model = keras.Model(inputs, outputs, name="residual_denoiser")
    return model
