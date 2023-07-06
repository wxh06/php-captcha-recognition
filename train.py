from datetime import datetime
from glob import glob

import tensorflow as tf
from keras import layers, models

DATA_DIR = "data"
LOG_DIR = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
H, W, C = 35, 90, 3  # height, width, 3 (RGB channels)
N_LABELS = 128
D = 4  # num_of_chars_per_image
EPOCHS = 16
PARALLELS = tf.data.AUTOTUNE
BATCH_SIZE = 1024
VALIDATION = 8


files = glob(f"{DATA_DIR}/*.tfrecords")

raw_captcha_dataset = tf.data.TFRecordDataset(files)

captcha_feature_description = {
    "phrase": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
    features = tf.io.parse_single_example(example_proto, captcha_feature_description)
    return (
        tf.cast(tf.io.parse_tensor(features["image"], tf.uint8), tf.float32) / 255.0,
        tf.one_hot(
            tf.strings.unicode_decode(features["phrase"], "ascii"),
            N_LABELS,
        ),
    )


parsed_captcha_dataset = (
    raw_captcha_dataset.map(
        _parse_image_function,
        num_parallel_calls=PARALLELS,
        deterministic=False,
    )
    .batch(BATCH_SIZE)
    .prefetch(1)
)

dataset_validation = parsed_captcha_dataset.take(VALIDATION)
dataset_train = parsed_captcha_dataset.skip(VALIDATION)


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


class EpochSaver(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        self.model.save(f"saved_models/{epoch}")


history = model.fit(
    dataset_train,
    epochs=EPOCHS,
    validation_data=dataset_validation,
    validation_steps=VALIDATION,
    callbacks=[
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1),
        EpochSaver(),
    ],
)

model.save("saved_model")
