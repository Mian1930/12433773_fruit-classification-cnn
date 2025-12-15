import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--class_indices", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)

    args = parser.parse_args()

    model = load_model(args.model, compile=False)

    with open(args.class_indices) as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    img = image.load_img(args.image, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    max_prob = np.max(preds)
    pred_idx = np.argmax(preds)

    if max_prob < args.threshold:
        print("❌ OOD DETECTED")
        print(f"Confidence too low: {max_prob:.2f}")
    else:
        print("✅ Prediction accepted")
        print(f"Class: {idx_to_class[pred_idx]}")
        print(f"Confidence: {max_prob:.2f}")


if __name__ == "__main__":
    main()
