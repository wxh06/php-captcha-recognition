"""
captcha-tensorflow
Copyright (c) 2017 Jackon Yang
https://github.com/JackonYang/captcha-tensorflow/blob/master/captcha-solver-tf2-4digits-AlexNet-98.8.ipynb
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras import layers, models
from keras.utils.np_utils import to_categorical
from PIL import Image


DATA_DIR = "data"
H, W, C = 35, 90, 3  # height, width, 3 (RGB channels)
N_LABELS = 128
D = 4  # num_per_image


def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        label, _ = filename.split("_")
        return label
    except Exception as e:
        print("error to parse %s. %s" % (filepath, e))
        return None, None


# create a pandas data frame of images, age, gender and race
files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
attributes = list(map(parse_filepath, files))

df = pd.DataFrame(attributes)
df["file"] = files
df.columns = ["label", "file"]
df = df.dropna()
print(df.head())


p = np.random.permutation(len(df))
train_up_to = int(len(df) * 0.7)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

print(
    "train count: %s, valid count: %s, test count: %s"
    % (len(train_idx), len(valid_idx), len(test_idx))
)


def get_data_generator(df, indices, for_training, batch_size=16):
    images, labels = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r["file"], r["label"]
            im = Image.open(file)
            #             im = im.resize((H, W))
            im = np.array(im) / 255.0
            images.append(np.array(im))
            labels.append(
                np.array(
                    [np.array(to_categorical(ord(i), N_LABELS)) for i in label]
                )
            )
            if len(images) >= batch_size:
                #                 print(np.array(images), np.array(labels))
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break


input_layer = tf.keras.Input(shape=(H, W, C))
x = layers.Conv2D(32, 3, activation="relu")(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
# x = layers.Dropout(0.5)(x)

x = layers.Dense(D * N_LABELS, activation="softmax")(x)
x = layers.Reshape((D, N_LABELS))(x)

model = models.Model(inputs=input_layer, outputs=x)

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()


batch_size = 64
valid_batch_size = 64
train_gen = get_data_generator(
    df, train_idx, for_training=True, batch_size=batch_size
)
valid_gen = get_data_generator(
    df, valid_idx, for_training=True, batch_size=valid_batch_size
)

history = model.fit(
    train_gen,
    steps_per_epoch=len(train_idx) // batch_size,
    epochs=3,
    validation_data=valid_gen,
    validation_steps=len(valid_idx) // valid_batch_size,
)


def plot_train_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].plot(history.history["accuracy"], label="Train accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Val accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_xlabel("Epochs")
    axes[1].legend()


plot_train_history(history)
plt.show()


# evaluate loss and accuracy in test dataset
test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
print(
    dict(
        zip(
            model.metrics_names,
            model.evaluate(test_gen, steps=len(test_idx) // 128),
        )
    )
)


model.save("saved_model/luogu_captcha")
