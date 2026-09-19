"""
Step 4: the CNN.

Input: normalised log-mel spectrogram, shape (64 mel bins, 94 frames, 1 channel)
       -> treated like a 1-channel image.

4 x [Conv2D -> BatchNorm -> ReLU -> MaxPool -> Dropout]
-> GlobalAveragePooling -> Dense -> Dropout -> Softmax (7 emotions)
"""
import keras
from keras import layers

from src import config


def conv_block(x, filters, dropout_rate, l2_strength):
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same", use_bias=False,
                      kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x


def build_cnn(input_shape=(config.N_MELS, config.N_FRAMES, 1), num_classes=config.NUM_CLASSES):
    l2_strength = 1e-4

    inputs = keras.Input(shape=input_shape, name="log_mel")

    x = conv_block(inputs, 32, dropout_rate=0.2, l2_strength=l2_strength)   # -> 32 x 47
    x = conv_block(x, 64, dropout_rate=0.2, l2_strength=l2_strength)        # -> 16 x 23
    x = conv_block(x, 128, dropout_rate=0.3, l2_strength=l2_strength)       # -> 8 x 11
    x = conv_block(x, 256, dropout_rate=0.3, l2_strength=l2_strength)       # -> 4 x 5

    # average every feature map to one number -> 256 values, works for any clip length
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    return keras.Model(inputs, outputs, name="ser_cnn")


def build_model(model_name):
    if model_name == "cnn":
        return build_cnn()
    raise ValueError(f"Unknown model: {model_name}")
