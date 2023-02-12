from concurrent.futures import ThreadPoolExecutor, wait, as_completed
from os import path
from subprocess import PIPE, Popen
from uuid import uuid4

import tensorflow as tf

PHP_BIN = "php"
DATA_DIR = "data"
PRINT = 64
BUFFER_SIZE = 65536

shutdown = False
count = 0


def generate():
    process = Popen([PHP_BIN, "gen.php"], stdout=PIPE)
    output = b""
    while True:
        output += process.stdout.read(BUFFER_SIZE)
        while True:
            eoi = output.find(b"\xff\xd9")
            if eoi < 0:
                break
            soi = output.find(b"\xff\xd8")
            eoi += 2

            phrase = output[:soi].decode("ascii")
            image = output[soi:eoi]
            output = output[eoi:]

            global count
            count += 1
            if not count % PRINT:
                print(f"\r{count:,}", end="", flush=True)

            yield phrase, image

        if shutdown:
            break


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def image_example(image_string: bytes, phrase: str):
    image = tf.image.decode_jpeg(image_string)
    feature = {
        "phrase": _bytes_feature(phrase.encode("ascii")),
        "image": _bytes_feature(tf.io.serialize_tensor(image)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def main():
    uuid = uuid4()
    with tf.io.TFRecordWriter(
        path.join(DATA_DIR, f"{uuid}.tfrecords")
    ) as writer:
        for phrase, image in generate():
            tf_example = image_example(image, phrase)
            writer.write(tf_example.SerializeToString())
    return uuid


if __name__ == "__main__":
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(main) for i in range(executor._max_workers)]
        try:
            wait(futures)
        except KeyboardInterrupt:
            shutdown = True
    print(f"\r{count:,}  ")
    for f in as_completed(futures):
        print(f.result())
