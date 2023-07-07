from os import path

import tensorflow as tf
from keras.models import load_model

MODEL_PATH = "saved_model"


model = load_model(path.join(path.dirname(__file__), MODEL_PATH))


def predict(image):
    prediction = model(
        tf.expand_dims(tf.cast(tf.image.decode_jpeg(image), tf.float32) / 255, axis=0)
    )
    return "".join(map(chr, map(int, tf.math.argmax(prediction, axis=-1)[0])))


if __name__ == "__main__":
    model.summary()
    print(predict(open("./captcha.jpeg", "rb").read()))
