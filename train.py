"""
captcha-tensorflow
Copyright (c) 2017 Jackon Yang
https://github.com/JackonYang/captcha-tensorflow/blob/master/captcha-solver-tf2-4digits-AlexNet-98.8.ipynb
"""

from datetime import datetime
from os import listdir, path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
from keras.utils.np_utils import to_categorical

DATA_DIR = "data"
LOG_DIR = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
H, W, C = 35, 90, 3  # height, width, 3 (RGB channels)
N_LABELS = 128
D = 4  # num_of_chars_per_image
EPOCHS = 4


# create a pandas data frame of images and labels
files = listdir(DATA_DIR)


p = np.random.permutation(len(files))
train_up_to = int(len(files) * 0.9375)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.9375)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

print(
    "train count: %s, valid count: %s, test count: %s"
    % (len(train_idx), len(valid_idx), len(test_idx))
)


def get_data_generator(files, indices, repeat=1):
    for _ in range(repeat):
        for i in indices:
            df = pd.read_pickle(path.join(DATA_DIR, files[i]))
            images = np.array([a for a in df["image"]]) / 255.0
            labels = np.array(
                [
                    [np.array(to_categorical(ord(i), N_LABELS)) for i in label.lower()]
                    for label in df["phrase"]
                ]
            )
            yield images, labels


input_layer = tf.keras.Input(shape=(H, W, C))
x = layers.Conv2D(32, 3, activation="relu")(input_layer)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D((2, 2))(x)

x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)

x = layers.Dense(D * N_LABELS, activation="softmax")(x)
x = layers.Reshape((D, N_LABELS))(x)

model = models.Model(inputs=input_layer, outputs=x)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


history = model.fit(
    get_data_generator(files, train_idx, EPOCHS),
    steps_per_epoch=len(train_idx),
    epochs=EPOCHS,
    validation_data=get_data_generator(files, valid_idx, EPOCHS),
    validation_steps=len(valid_idx),
    callbacks=[tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)],
)


# evaluate loss and accuracy in test dataset
test_gen = get_data_generator(files, test_idx)
print(
    dict(
        zip(
            model.metrics_names,
            model.evaluate(test_gen, steps=len(test_idx)),
        )
    )
)


model.save("saved_model/luogu_captcha")
