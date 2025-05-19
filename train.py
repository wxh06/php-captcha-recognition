from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, HEIGHT, LENGTH, WIDTH
from model import build_model

# Label
NUM_CHARS = LENGTH
CHARSET = list("abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Image
SIZE = (WIDTH, HEIGHT)
CHANNELS = 3
SHAPE = (*SIZE, CHANNELS)

# Path to the data directory
data_dir = Path("./data/")

# Get list of all the images
files = list(map(str, data_dir.glob("*.tfrecords")))

print("Number of TFRecord files found: ", len(files))
print("Number of unique characters: ", len(CHARSET))
print("Characters present: ", CHARSET)

validation_size = 256


# Mapping characters to integers
string_lookup = tf.keras.layers.StringLookup(
    vocabulary=CHARSET, mask_token=None, num_oov_indices=0
)

feature_description = {
    "phrase": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def parse_tfrecord(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    img = tf.io.decode_jpeg(parsed["image"], channels=3)
    img = tf.image.resize(img, SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)

    label_str = parsed["phrase"]
    labels = tf.strings.unicode_split(label_str, "UTF-8")
    labels = string_lookup(labels)
    labels = tf.ensure_shape(labels, [NUM_CHARS])

    return img, {f"char_{i}": labels[i] for i in range(NUM_CHARS)}


dataset = tf.data.TFRecordDataset(files)

train_dataset = dataset.skip(validation_size)
train_dataset = (
    train_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = dataset.take(validation_size)
validation_dataset = (
    validation_dataset.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# 使用示例
model = build_model(SHAPE, NUM_CHARS, len(CHARSET))

# 编译配置（只需针对数字输出进行优化）
model.compile(
    optimizer="adam",
    loss={f"char_{i}": "sparse_categorical_crossentropy" for i in range(NUM_CHARS)},
    metrics={f"char_{i}": "accuracy" for i in range(NUM_CHARS)},
)
model.summary()


# TODO restore epoch count.
epochs = 10
early_stopping_patience = 3
# Add early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)
