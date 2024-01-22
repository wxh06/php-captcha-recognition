import sys
from os import path
from subprocess import PIPE, Popen

import tensorflow as tf

PHP_BINARY = "php"
DATA_DIR = "data"
CHUNK_SIZE = 65536


def generate():
    process = Popen([PHP_BINARY, "generate.php"], stdout=PIPE)
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
def image_example(image_string: bytes, phrase: str):
    image = tf.image.decode_jpeg(image_string)
    feature = {
        "image": _bytes_feature(tf.io.serialize_tensor(image)),
        "label": _bytes_feature(phrase.encode("ascii")),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write(name: str):
    with tf.io.TFRecordWriter(path.join(DATA_DIR, f"{name}.tfrecords")) as writer:
        for phrase, image in generate():
            tf_example = image_example(image, phrase)
            writer.write(tf_example.SerializeToString())


if __name__ == "__main__":
    try:
        write(sys.argv[1])
    except KeyboardInterrupt:
        pass
