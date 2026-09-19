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


def build_cnn_lstm(input_shape=(config.N_MELS, config.N_FRAMES, 1), num_classes=config.NUM_CLASSES):
    """
    Step 6 extension: CNN + bidirectional LSTM + attention.

    The CNN learns local patterns (pitch, energy, formants). Instead of averaging
    over time, the LSTM reads the CNN output frame by frame so it can model how
    the emotion develops across the sentence, and an attention layer learns
    which frames matter most (e.g. the stressed word).
    """
    l2_strength = 1e-4

    inputs = keras.Input(shape=input_shape, name="log_mel")

    x = conv_block(inputs, 32, dropout_rate=0.2, l2_strength=l2_strength)   # -> 32 x 47 x 32
    x = conv_block(x, 64, dropout_rate=0.2, l2_strength=l2_strength)        # -> 16 x 23 x 64
    x = conv_block(x, 128, dropout_rate=0.3, l2_strength=l2_strength)       # -> 8 x 11 x 128

    # (freq, time, channels) -> (time, freq * channels): one feature vector per time step
    x = layers.Permute((2, 1, 3))(x)                                        # -> 11 x 8 x 128
    time_steps = x.shape[1]
    x = layers.Reshape((time_steps, x.shape[2] * x.shape[3]))(x)            # -> 11 x 1024
    x = layers.Dense(128, activation="relu")(x)                             # shrink each step
    x = layers.Dropout(0.3)(x)

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)     # -> 11 x 128

    # attention pooling: a score per time step -> softmax -> weighted sum of the steps
    scores = layers.Dense(1)(x)                                             # -> 11 x 1
    weights = layers.Softmax(axis=1, name="attention_weights")(scores)
    x = layers.Multiply()([x, weights])
    x = layers.GlobalAveragePooling1D()(x)   # mean of weighted steps = weighted sum / 11

    x = layers.Dense(128, activation="relu", kernel_regularizer=keras.regularizers.l2(l2_strength))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="emotion")(x)

    return keras.Model(inputs, outputs, name="ser_cnn_lstm")


def build_model(model_name):
    if model_name == "cnn":
        return build_cnn()
    if model_name == "cnn_lstm":
        return build_cnn_lstm()
    raise ValueError(f"Unknown model: {model_name}")
