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
EPOCHS = 16


# create a pandas data frame of images and labels
files = listdir(DATA_DIR)

np.random.shuffle(files)
split = int(len(files) / 32)
files_train = files[: -split * 2]
files_validation = files[-split * 2 : -split]
files_test = files[-split:]


def get_data_generator(files, repeat=False):
    while True:
        for file in files:
            df = pd.read_pickle(path.join(DATA_DIR, file))
            images = np.array([a for a in df["image"]]) / 255.0
            labels = np.array(
                [
                    [np.array(to_categorical(ord(i), N_LABELS)) for i in label]
                    for label in df["phrase"]
                ]
            )
            yield images, labels
        if not repeat:
            return


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

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)
model.summary()


class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        self.model.save("saved_model/model_{}".format(epoch))


history = model.fit(
    get_data_generator(files_train, True),
    steps_per_epoch=len(files_train),
    epochs=EPOCHS,
    validation_data=get_data_generator(files_validation, True),
    validation_steps=len(files_validation),
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
        EpochSaver(),
    ],
)

# evaluate loss and accuracy in test dataset
print(
    dict(
        zip(
            model.metrics_names,
            model.evaluate(
                get_data_generator(files_test), steps=len(files_test)
            ),
        )
    )
)


model.save("saved_model/luogu_captcha")
