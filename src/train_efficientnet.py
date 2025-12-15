import os
import json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from data_loader_fruits360 import create_generators


def main():
    tf.keras.backend.clear_session()

    dataset_root = "/home/jovyan/fruits360"

    train_gen, val_gen, test_gen, class_indices = create_generators(
        dataset_root
    )

    base = EfficientNetB0(
        weights=None,                 # ✅ NO IMAGENET
        include_top=False,
        input_shape=(224, 224, 3)     # ✅ RGB
    )

    x = GlobalAveragePooling2D()(base.output)
    outputs = Dense(len(class_indices), activation="softmax")(x)

    model = Model(base.input, outputs)

    model.compile(
        optimizer=Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10
    )

    os.makedirs("outputs_efficientnet", exist_ok=True)
    model.save("outputs_efficientnet/best_model_efficientnet.h5")

    with open("outputs_efficientnet/class_indices.json", "w") as f:
        json.dump(class_indices, f)

    print("✅ Training complete")


if __name__ == "__main__":
    main()
