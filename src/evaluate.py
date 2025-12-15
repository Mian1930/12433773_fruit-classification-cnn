import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
from data_loader import create_generators
import seaborn as sns

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--test_dir', required=True)
    p.add_argument('--class_indices', required=True)
    p.add_argument('--out', default='evaluation')
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load class indices
    with open(args.class_indices) as f:
        class_indices = json.load(f)
    idx2class = {v: k for k, v in class_indices.items()}
    class_names = [idx2class[i] for i in range(len(idx2class))]

    # Load model (IMPORTANT FIX)
    model = load_model(args.model, compile=False)

    _, _, test_gen, _ = create_generators(
        args.test_dir,
        batch_size=32,
        shuffle=False
    )

    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    # Classification report
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png")
    plt.close()

    print(f"✔ Report saved to {report_path}")
    print(f"✔ Confusion matrix saved")

if __name__ == "__main__":
    main()
