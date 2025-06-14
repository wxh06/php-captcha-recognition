from datetime import datetime
from pathlib import Path

import tensorflow as tf

from config import BATCH_SIZE, EPOCHS, HEIGHT, LENGTH, WIDTH
from model import build_model
from utils import normalize_label, string_lookup, vocabulary

# Label
NUM_CHARS = LENGTH

# Image
SIZE = (HEIGHT, WIDTH)
CHANNELS = 3
SHAPE = (*SIZE, CHANNELS)

# Path to the data directory
data_dir = Path("./data/")

files = list(map(str, data_dir.glob(f"{LENGTH}-{WIDTH}x{HEIGHT}-*.tfrecords")))

time = datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = f"checkpoints/{time}/" + "cp-{epoch:04d}.weights.h5"
log_dir = f"logs/fit/{time}"

print("Number of TFRecord files found: ", len(files))
print("Number of unique characters: ", len(vocabulary))
print("Characters present: ", vocabulary)

validation_size = 65536


feature_description = {
    "phrase": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def parse_tfrecord(example_proto):
    parsed = tf.io.parse_single_example(example_proto, feature_description)

    img = tf.io.decode_jpeg(parsed["image"], channels=3)
    img = tf.image.resize(img, SIZE)

    label_str = normalize_label(parsed["phrase"])
    labels = tf.strings.unicode_split(label_str, "UTF-8")
    labels = string_lookup(labels)
    labels = tf.ensure_shape(labels, [NUM_CHARS])

    return img, {
        **{f"char_{i}": labels[i] for i in range(NUM_CHARS)},
        "phrase": labels,
    }


dataset = tf.data.TFRecordDataset(files).map(
    parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = dataset.skip(validation_size)
train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

validation_dataset = dataset.take(validation_size)
validation_dataset = validation_dataset.batch(1024).prefetch(tf.data.AUTOTUNE)

model = build_model(SHAPE, NUM_CHARS, len(vocabulary))


def exact_match_accuracy(y_true, y_pred):
    exact_matches = tf.reduce_all(tf.equal(y_true, y_pred), axis=1)
    exact_matches = tf.cast(exact_matches, tf.float32)
    return tf.reduce_mean(exact_matches)


model.compile(
    optimizer="adam",
    loss={
        **{f"char_{i}": "sparse_categorical_crossentropy" for i in range(NUM_CHARS)},
        "phrase": None,
    },
    metrics={
        **{f"char_{i}": "accuracy" for i in range(NUM_CHARS)},
        "phrase": exact_match_accuracy,
    },
)
model.summary()


cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_phrase_exact_match_accuracy",
    patience=5,
    mode="max",
    restore_best_weights=True,
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=3
)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[cp_callback, early_stopping, reduce_lr, tensorboard_callback],
)

model.export(f"models/{LENGTH}-{WIDTH}x{HEIGHT}-{time}")
