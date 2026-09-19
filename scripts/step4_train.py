"""
STEP 4 - Train the CNN

    python scripts/step4_train.py                 # CNN (default)
    python scripts/step4_train.py --epochs 80

Training data   = original train clips + augmented copies (step 3)
                  + SpecAugment on every batch
Validation data = actors 17-20, never augmented
Imbalance       = class weights ("neutral" has 1.5x clips because calm was merged into it)
Callbacks       = early stopping, learning-rate reduction, best-model checkpoint

Saves:
    models/{model}.keras                  best model (highest validation accuracy)
    models/norm_stats.npz                 per-mel-bin mean / std from the training set
    reports/{model}_history.csv           loss / accuracy per epoch
    reports/figures/06_{model}_training_curves.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from src import config
from src.data_generator import SpecAugmentGenerator
from src.features import compute_norm_stats, normalize
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def load_training_data():
    train_mels = np.load(config.FEATURES_DIR / "train_mels.npy")
    train_labels = np.load(config.FEATURES_DIR / "train_labels.npy")
    aug_mels = np.load(config.FEATURES_DIR / "train_aug_mels.npy")
    aug_labels = np.load(config.FEATURES_DIR / "train_aug_labels.npy")
    val_mels = np.load(config.FEATURES_DIR / "val_mels.npy")
    val_labels = np.load(config.FEATURES_DIR / "val_labels.npy")

    # normalisation statistics come from the clean training clips only
    mean, std = compute_norm_stats(train_mels)

    all_train_mels = np.concatenate([train_mels, aug_mels], axis=0)
    all_train_labels = np.concatenate([train_labels, aug_labels], axis=0)

    all_train_mels = normalize(all_train_mels, mean, std)
    val_mels = normalize(val_mels, mean, std)[..., np.newaxis]   # add channel dimension

    return all_train_mels, all_train_labels, val_mels, val_labels, mean, std


def get_class_weights(labels):
    classes = np.arange(config.NUM_CLASSES)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    class_weights = {}
    for class_index in classes:
        class_weights[int(class_index)] = float(weights[class_index])
    return class_weights


def plot_history(history, model_name):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.suptitle(f"Training curves - {model_name}")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"06_{model_name}_training_curves.png", dpi=110)
    plt.close()


def main():
    args = parse_args()
    keras.utils.set_random_seed(config.RANDOM_SEED)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    train_x, train_y, val_x, val_y, mean, std = load_training_data()
    np.savez(config.MODELS_DIR / "norm_stats.npz", mean=mean, std=std)
    print(f"Training spectrograms: {len(train_x)}   Validation clips: {len(val_x)}")

    class_weights = get_class_weights(train_y)
    print("Class weights:")
    for class_index in class_weights:
        print(f"  {config.EMOTIONS[class_index]:>9}: {class_weights[class_index]:.2f}")

    train_generator = SpecAugmentGenerator(train_x, train_y, batch_size=args.batch_size,
                                           seed=config.RANDOM_SEED)

    model = build_model(args.model)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model_path = config.MODELS_DIR / f"{args.model}.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(model_path, monitor="val_accuracy", mode="max",
                                        save_best_only=True, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                          min_lr=1e-5, verbose=1),
    ]

    history = model.fit(
        train_generator,
        validation_data=(val_x, val_y),
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )

    history_table = pd.DataFrame(history.history)
    history_table.index.name = "epoch"
    history_table.to_csv(config.REPORTS_DIR / f"{args.model}_history.csv")
    plot_history(history.history, args.model)

    best_epoch = int(np.argmax(history.history["val_accuracy"]))
    print(f"\nBest validation accuracy: {history.history['val_accuracy'][best_epoch]:.4f} "
          f"(epoch {best_epoch + 1})")
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
