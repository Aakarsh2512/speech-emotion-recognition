"""
STEP 5 - Evaluate on the held-out TEST actors (21-24)

    python scripts/step5_evaluate.py              # evaluates models/cnn.keras
    python scripts/step5_evaluate.py --model cnn_lstm

These 4 actors were never seen during training or model selection,
so this is an honest estimate of performance on NEW speakers.

Reports: accuracy, macro F1, per-class precision / recall / F1,
confusion matrix, accuracy per actor and per gender, most common mistakes.

Saves:
    reports/{model}_results.json
    reports/{model}_classification_report.txt
    reports/figures/07_{model}_confusion_matrix.png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src import config
from src.features import normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn")
    return parser.parse_args()


def plot_confusion_matrix(true_labels, predicted_labels, model_name):
    counts = confusion_matrix(true_labels, predicted_labels, labels=np.arange(config.NUM_CLASSES))
    # divide each row by its total -> "what % of real X was predicted as Y"
    row_totals = counts.sum(axis=1, keepdims=True)
    percentages = counts / row_totals

    figure, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS, ax=axes[0])
    axes[0].set_title("Counts")

    sns.heatmap(percentages, annot=True, fmt=".0%", cmap="Blues", cbar=False, vmin=0, vmax=1,
                xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS, ax=axes[1])
    axes[1].set_title("Row-normalised (= recall on the diagonal)")

    for axis in axes:
        axis.set_xlabel("Predicted emotion")
        axis.set_ylabel("True emotion")

    plt.suptitle(f"Confusion matrix - {model_name} - unseen test actors 21-24")
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f"07_{model_name}_confusion_matrix.png", dpi=110)
    plt.close()
    return counts


def most_common_mistakes(counts, how_many=5):
    mistakes = []
    for true_index in range(config.NUM_CLASSES):
        for predicted_index in range(config.NUM_CLASSES):
            if true_index != predicted_index and counts[true_index][predicted_index] > 0:
                mistakes.append((int(counts[true_index][predicted_index]),
                                 config.EMOTIONS[true_index], config.EMOTIONS[predicted_index]))
    mistakes.sort(reverse=True)
    return mistakes[:how_many]


def main():
    args = parse_args()

    model = keras.models.load_model(config.MODELS_DIR / f"{args.model}.keras")
    stats = np.load(config.MODELS_DIR / "norm_stats.npz")

    test_mels = np.load(config.FEATURES_DIR / "test_mels.npy")
    test_labels = np.load(config.FEATURES_DIR / "test_labels.npy")
    test_actors = np.load(config.FEATURES_DIR / "test_actors.npy")

    test_x = normalize(test_mels, stats["mean"], stats["std"])[..., np.newaxis]
    probabilities = model.predict(test_x, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(test_labels, predictions)
    macro_f1 = f1_score(test_labels, predictions, average="macro")
    weighted_f1 = f1_score(test_labels, predictions, average="weighted")
    report_text = classification_report(test_labels, predictions, target_names=config.EMOTIONS, digits=3)
    report_dict = classification_report(test_labels, predictions, target_names=config.EMOTIONS,
                                        output_dict=True)

    print(f"\n===== {args.model} on unseen test actors {config.TEST_ACTORS} =====")
    print(f"Clips:        {len(test_labels)}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Macro F1:     {macro_f1:.4f}")
    print(f"Weighted F1:  {weighted_f1:.4f}")
    print("\n" + report_text)

    counts = plot_confusion_matrix(test_labels, predictions, args.model)

    print("Most common mistakes (true -> predicted):")
    mistakes = most_common_mistakes(counts)
    for count, true_emotion, predicted_emotion in mistakes:
        print(f"  {true_emotion:>9} -> {predicted_emotion:<9} {count} clips")

    print("\nAccuracy per test actor:")
    per_actor = {}
    for actor in config.TEST_ACTORS:
        mask = test_actors == actor
        actor_accuracy = accuracy_score(test_labels[mask], predictions[mask])
        if actor % 2 == 1:
            gender = "male"
        else:
            gender = "female"
        per_actor[str(actor)] = round(float(actor_accuracy), 4)
        print(f"  actor {actor} ({gender:>6}): {actor_accuracy:.4f}")

    results = {
        "model": args.model,
        "test_actors": config.TEST_ACTORS,
        "num_test_clips": int(len(test_labels)),
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": {},
        "per_actor_accuracy": per_actor,
        "top_confusions": [],
    }
    for count, true_emotion, predicted_emotion in mistakes:
        results["top_confusions"].append({"true": true_emotion, "predicted": predicted_emotion,
                                          "count": count})
    for emotion in config.EMOTIONS:
        results["per_class"][emotion] = {
            "precision": round(report_dict[emotion]["precision"], 4),
            "recall": round(report_dict[emotion]["recall"], 4),
            "f1": round(report_dict[emotion]["f1-score"], 4),
            "support": int(report_dict[emotion]["support"]),
        }

    with open(config.REPORTS_DIR / f"{args.model}_results.json", "w") as results_file:
        json.dump(results, results_file, indent=2)
    with open(config.REPORTS_DIR / f"{args.model}_classification_report.txt", "w") as report_file:
        report_file.write(f"{args.model} - unseen test actors {config.TEST_ACTORS}\n")
        report_file.write(f"accuracy {accuracy:.4f} | macro F1 {macro_f1:.4f}\n\n")
        report_file.write(report_text)

    print(f"\nResults saved to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
