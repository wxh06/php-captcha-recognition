import sys
from pathlib import Path
from subprocess import PIPE, Popen

import tensorflow as tf

from config import HEIGHT, LENGTH, WIDTH

PHP_BINARY = "php"
DATA_DIR = Path("./data/")
CHUNK_SIZE = 65536


def generate():
    process = Popen(
        [PHP_BINARY, "generate.php", str(LENGTH), str(WIDTH), str(HEIGHT)],
        stdout=PIPE,
    )
    output = b""
    while True:
        output += process.stdout.read(CHUNK_SIZE)
        while True:
            eoi = output.find(b"\xff\xd9")
            if eoi < 0:
                break
            soi = output.find(b"\xff\xd8")
            eoi += 2

            phrase = output[:soi].decode("ascii")
            image = output[soi:eoi]
            output = output[eoi:]

            yield phrase, image


# https://www.tensorflow.org/tutorials/load_data/tfrecord#data_types_for_tftrainexample
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# https://www.tensorflow.org/tutorials/load_data/tfrecord#write_the_tfrecord_file
def captcha_example(image_raw: bytes, phrase: str):
    feature = {
        "phrase": _bytes_feature(phrase.encode("ascii")),
        "image": _bytes_feature(image_raw),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write(name: str):
    with tf.io.TFRecordWriter(
        str(DATA_DIR / f"{LENGTH}-{WIDTH}x{HEIGHT}-{name}.tfrecords")
    ) as writer:
        for phrase, image in generate():
            tf_example = captcha_example(image, phrase)
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    try:
        uuid = sys.argv[1]
    except IndexError:
        from uuid import uuid4

        uuid = str(uuid4())

    try:
        write(uuid)
    except KeyboardInterrupt:
        pass
