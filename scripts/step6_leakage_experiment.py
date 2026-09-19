"""
STEP 6 (extension) - Why the speaker-independent split matters

    python scripts/step6_leakage_experiment.py

Trains the SAME CNN with the SAME pipeline as step 4 (offline augmentation,
SpecAugment, class weights, early stopping). Only the split changes:

    A. speaker-independent : train actors 1-16, val 17-20, test 21-24
                             -> result already in reports/cnn_results.json (step 5)
    B. random              : the 1440 clips shuffled into splits of the same sizes
                             (960 / 240 / 240), so every test actor was also heard in
                             training, often saying the exact same sentence

Saves:
    reports/leakage_experiment.json
    reports/figures/08_leakage_experiment.png
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from scripts.step3_augment import AUGMENTED_COPIES_PER_CLIP, make_augmented_versions
from scripts.step4_train import get_class_weights
from src import config
from src.data_generator import SpecAugmentGenerator
from src.features import compute_norm_stats, log_mel_spectrogram, normalize
from src.model import build_cnn


def load_all_clips():
    """Clean spectrograms + labels of all 1440 clips, in the order train, val, test."""
    mels = []
    labels = []
    for split in ["train", "val", "test"]:
        mels.append(np.load(config.FEATURES_DIR / f"{split}_mels.npy"))
        labels.append(np.load(config.FEATURES_DIR / f"{split}_labels.npy"))
    return np.concatenate(mels), np.concatenate(labels)


def load_all_augmented():
    """
    The 3 augmented copies of every clip, same order as load_all_clips().
    Train copies come from step 3; val/test clips are augmented here once and cached,
    because in the random split some of them end up in the training set.
    """
    train_aug = np.load(config.FEATURES_DIR / "train_aug_mels.npy")

    cache_path = config.FEATURES_DIR / "valtest_aug_mels.npy"
    if cache_path.exists():
        valtest_aug = np.load(cache_path)
    else:
        rng = np.random.default_rng(config.RANDOM_SEED + 1)
        valtest_waves = np.concatenate([np.load(config.FEATURES_DIR / "val_waves.npy"),
                                        np.load(config.FEATURES_DIR / "test_waves.npy")])
        valtest_aug = np.zeros((len(valtest_waves) * AUGMENTED_COPIES_PER_CLIP,
                                config.N_MELS, config.N_FRAMES), dtype=np.float32)
        output_index = 0
        for clip_index in tqdm(range(len(valtest_waves)), desc="Augmenting val/test clips"):
            for augmented_audio in make_augmented_versions(valtest_waves[clip_index], rng):
                valtest_aug[output_index] = log_mel_spectrogram(augmented_audio)
                output_index += 1
        np.save(cache_path, valtest_aug)

    all_aug = np.concatenate([train_aug, valtest_aug])
    # reshape so that all_aug[clip] gives the 3 copies of that clip
    return all_aug.reshape(-1, AUGMENTED_COPIES_PER_CLIP, config.N_MELS, config.N_FRAMES)


def main():
    keras.utils.set_random_seed(config.RANDOM_SEED)

    all_mels, all_labels = load_all_clips()
    all_aug = load_all_augmented()

    # random split with exactly the same sizes as the speaker-independent one
    num_val = len(np.load(config.FEATURES_DIR / "val_labels.npy"))
    num_test = len(np.load(config.FEATURES_DIR / "test_labels.npy"))
    clip_ids = np.arange(len(all_labels))
    rest_ids, test_ids = train_test_split(clip_ids, test_size=num_test, stratify=all_labels,
                                          random_state=config.RANDOM_SEED)
    train_ids, val_ids = train_test_split(rest_ids, test_size=num_val, stratify=all_labels[rest_ids],
                                          random_state=config.RANDOM_SEED)
    print(f"Random split: train {len(train_ids)}, val {len(val_ids)}, test {len(test_ids)}")

    # training set = clean train clips + their augmented copies (exactly like step 4)
    train_clean = all_mels[train_ids]
    train_aug = all_aug[train_ids].reshape(-1, config.N_MELS, config.N_FRAMES)
    train_aug_labels = np.repeat(all_labels[train_ids], AUGMENTED_COPIES_PER_CLIP)

    mean, std = compute_norm_stats(train_clean)
    train_x = normalize(np.concatenate([train_clean, train_aug]), mean, std)
    train_y = np.concatenate([all_labels[train_ids], train_aug_labels])
    val_x = normalize(all_mels[val_ids], mean, std)[..., np.newaxis]
    val_y = all_labels[val_ids]
    test_x = normalize(all_mels[test_ids], mean, std)[..., np.newaxis]
    test_y = all_labels[test_ids]

    model = build_cnn()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=15,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                                          min_lr=1e-5, verbose=1),
    ]
    model.fit(SpecAugmentGenerator(train_x, train_y, batch_size=32, seed=config.RANDOM_SEED),
              validation_data=(val_x, val_y), epochs=60,
              class_weight=get_class_weights(train_y), callbacks=callbacks, verbose=2)

    predictions = np.argmax(model.predict(test_x, verbose=0), axis=1)
    random_accuracy = float(accuracy_score(test_y, predictions))
    random_f1 = float(f1_score(test_y, predictions, average="macro"))

    with open(config.REPORTS_DIR / "cnn_results.json") as results_file:
        speaker_results = json.load(results_file)
    speaker_accuracy = speaker_results["accuracy"]
    speaker_f1 = speaker_results["macro_f1"]

    results = {
        "note": "Same CNN and training pipeline as step 4; only the split differs",
        "speaker_independent": {"accuracy": speaker_accuracy, "macro_f1": speaker_f1},
        "random_split": {"accuracy": round(random_accuracy, 4), "macro_f1": round(random_f1, 4)},
        "inflation_accuracy_points": round((random_accuracy - speaker_accuracy) * 100, 1),
    }
    with open(config.REPORTS_DIR / "leakage_experiment.json", "w") as results_file:
        json.dump(results, results_file, indent=2)

    names = ["Speaker-independent\n(honest)", "Random split\n(speaker leakage)"]
    accuracies = [speaker_accuracy, random_accuracy]
    plt.figure(figsize=(6, 4.5))
    bars = plt.bar(names, accuracies, color=["#4C72B0", "#DD8452"], width=0.5)
    for bar_index in range(len(bars)):
        bar = bars[bar_index]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{accuracies[bar_index]:.1%}", ha="center", fontsize=11)
    plt.ylim(0, 1)
    plt.ylabel("Test accuracy")
    plt.title("Same model and pipeline, different split")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / "08_leakage_experiment.png", dpi=110)
    plt.close()

    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
