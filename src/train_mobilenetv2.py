import os
import json
import argparse
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

from data_loader_fruits360 import create_generators


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/home/jovyan/fruits360")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--out_dir", default="outputs_mobilenetv2")
    return p.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_dir = os.path.join(args.data_dir, "Training")
    test_dir = os.path.join(args.data_dir, "Test")

    # ⚠ Fruits360 has no validation → use validation_split
    train_gen, val_gen, test_gen, num_classes = create_generators(
        train_dir=train_dir,
        valid_dir=train_dir,
        test_dir=test_dir,
        img_size=(224, 224),
        batch_size=args.batch_size,
        model_type="mobilenetv2"
    )

    # 🔒 Freeze base model
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model_path = os.path.join(
        out_dir, f"best_model_mobilenetv2_{timestamp}.h5"
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True),
        CSVLogger(os.path.join(out_dir, f"training_log_{timestamp}.csv")),
        EarlyStopping(patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Save class indices
    with open(os.path.join(out_dir, "class_indices.json"), "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # 📈 Plot learning curves
    plt.figure()
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.legend()
    plt.title("Accuracy")
    plt.savefig(os.path.join(out_dir, "accuracy.png"))

    plt.figure()
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.legend()
    plt.title("Loss")
    plt.savefig(os.path.join(out_dir, "loss.png"))

    print(f"\nTraining finished.")
    print(f"Best model saved at: {model_path}")


if __name__ == "__main__":
    main()
