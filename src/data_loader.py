from pathlib import Path
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


def create_generators(
    dataset_dir,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    val_split=0.15,
    test_split=0.15,
    seed=SEED,
    shuffle=True
):
   

    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_dir}")

    # --------------------------------------------------
    # Collect all images + labels
    # --------------------------------------------------
    images = []
    labels = []

    class_names = sorted([p.name for p in dataset_dir.iterdir() if p.is_dir()])
    for cls in class_names:
        for img in (dataset_dir / cls).glob("*"):
            images.append(str(img))
            labels.append(cls)

    if len(images) == 0:
        raise RuntimeError("No images found in dataset directory")

    images = np.array(images)
    labels = np.array(labels)

    # --------------------------------------------------
    # Shuffle
    # --------------------------------------------------
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(images))
    images, labels = images[idx], labels[idx]

    # --------------------------------------------------
    # Split sizes
    # --------------------------------------------------
    n_total = len(images)
    n_test = int(n_total * test_split)
    n_val = int(n_total * val_split)

    X_test, y_test = images[:n_test], labels[:n_test]
    X_val, y_val = images[n_test:n_test+n_val], labels[n_test:n_test+n_val]
    X_train, y_train = images[n_test+n_val:], labels[n_test+n_val:]

    # --------------------------------------------------
    # DataFrames
    # --------------------------------------------------
    df_train = pd.DataFrame({"filename": X_train, "class": y_train})
    df_val   = pd.DataFrame({"filename": X_val,   "class": y_val})
    df_test  = pd.DataFrame({"filename": X_test,  "class": y_test})

    # --------------------------------------------------
    # Generators
    # --------------------------------------------------
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        df_train,
        x_col="filename",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = test_datagen.flow_from_dataframe(
        df_val,
        x_col="filename",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_dataframe(
        df_test,
        x_col="filename",
        y_col="class",
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen, train_gen.class_indices
