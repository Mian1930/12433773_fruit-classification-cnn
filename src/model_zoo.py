# model_zoo.py
import tensorflow as tf
from tensorflow.keras import layers, models

def build_small_cnn(input_shape=(128,128,3), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def build_transfer_model(base_name='MobileNetV2',
                         input_shape=(128,128,3),
                         num_classes=10,
                         train_base=False):
    if base_name == 'MobileNetV2':
        base = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    elif base_name == 'ResNet50':
        base = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    else:
        raise ValueError("Unsupported base")

    base.trainable = train_base  # False for feature extraction

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) if base_name=='MobileNetV2' else tf.keras.applications.resnet.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model
