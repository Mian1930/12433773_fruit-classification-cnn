# train.py
import os
import argparse
import json
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from data_loader import create_generators
from model_zoo import build_small_cnn, build_transfer_model
import numpy as np
from sklearn.metrics import classification_report


def try_init_wandb(use_wandb, config):
    if not use_wandb:
        return None
    try:
        import wandb
        from wandb.keras import WandbCallback
    except Exception:
        print("W&B not available. Running without it.")
        return None
    wandb.init(project="fruit-classification-12433773", config=config)
    return WandbCallback()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model', type=str, default='mobilenetv2',
                        choices=['small_cnn', 'mobilenetv2', 'resnet50'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--train_base', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train_gen, val_gen, test_gen, class_indices = create_generators(
        args.data_dir, batch_size=args.batch_size
    )

    if args.model == 'small_cnn':
        model = build_small_cnn((128, 128, 3), len(class_indices))
    elif args.model == 'mobilenetv2':
        model = build_transfer_model('MobileNetV2', (128, 128, 3), len(class_indices), train_base=args.train_base)
    else:
        model = build_transfer_model('ResNet50', (128, 128, 3), len(class_indices), train_base=args.train_base)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    ckpt_name = "best_model_" + args.model + "_" + timestamp + ".h5"
    ckpt_path = os.path.join(args.out_dir, ckpt_name)

    callbacks = [
        ModelCheckpoint(filepath=ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        CSVLogger(os.path.join(args.out_dir, "training_log_" + args.model + "_" + timestamp + ".csv"))
    ]

    wandb_cb = try_init_wandb(args.use_wandb, {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    })
    if wandb_cb is not None:
        callbacks.append(wandb_cb)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    test_steps = max(1, test_gen.samples // test_gen.batch_size + 1)
    preds = model.predict(test_gen, steps=test_steps)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    ordered_classes = sorted(class_indices.items(), key=lambda x: x[1])
    target_names = [name for name, _ in ordered_classes]

    report = classification_report(y_true, y_pred, target_names=target_names)
    print(report)

    report_filename = "classification_report_" + args.model + "_" + timestamp + ".txt"
    report_path = os.path.join(args.out_dir, report_filename)

    f = open(report_path, "w")
    f.write(report)
    f.close()

    mapping_path = os.path.join(args.out_dir, "class_indices.json")
    f = open(mapping_path, "w")
    json.dump(class_indices, f)
    f.close()

    print("Training finished.")
    print("Best model saved at:", ckpt_path)
    print("Report saved at:", report_path)


if __name__ == "__main__":
    main()
