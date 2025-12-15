from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path


def create_generators(dataset_root, img_size=(224, 224), batch_size=32):
    dataset_root = Path(dataset_root)

    train_dir = dataset_root / "Training"
    test_dir = dataset_root / "Test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    # ✅ RGB 
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        subset="training",
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        subset="validation",
        shuffle=False
    )

    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        color_mode="rgb",
        shuffle=False
    )

    return train_gen, val_gen, test_gen, train_gen.class_indices
