from pathlib import Path

import keras
import tensorflow as tf
from keras import layers

from captcha_ocr import build_model

# Path to the data directory
data_dir = Path("./data/")

# Get list of all the images
files = list(data_dir.glob("*.tfrecords"))
characters = sorted("abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("Number of TFRecord files found: ", len(files))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 90
img_height = 35

validation_size = 256


# Mapping characters to integers
char_to_num = layers.StringLookup(vocabulary=characters, mask_token=None)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)

captcha_feature_description = {
    "phrase": tf.io.FixedLenFeature([], tf.string),
    "image": tf.io.FixedLenFeature([], tf.string),
}


def encode_single_sample(example_proto):
    # 1. Read image
    features = tf.io.parse_single_example(example_proto, captcha_feature_description)
    # 2. Decode
    img = tf.io.parse_tensor(features["image"], tf.uint8)
    # 3. Convert to float32 in [0, 1] range
    img = tf.image.convert_image_dtype(img, tf.float32)
    # 4. Resize to the desired size
    # img = tf.image.resize(img, [img_height, img_width])
    # 5. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # 6. Map the characters in label to numbers
    label = char_to_num(tf.strings.upper(tf.strings.bytes_split(features["phrase"])))
    # 7. Return a dict as our model is expecting two inputs
    return {"image": img, "label": label}


dataset = tf.data.TFRecordDataset(files)

train_dataset = dataset.skip(validation_size)
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = dataset.take(validation_size)
validation_dataset = (
    validation_dataset.map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# Get the model
model = build_model(img_width, img_height, 3, num_to_char)
model.summary()


# TODO restore epoch count.
epochs = 10
early_stopping_patience = 3
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
)
