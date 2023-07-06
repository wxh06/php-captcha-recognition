from os import path

import numpy as np  # linear algebra
import tensorflow as tf
from keras.models import load_model
from PIL import Image

MODEL_PATH = "saved_model"


model = load_model(path.join(path.dirname(__file__), MODEL_PATH))


def predict(image):
    im = Image.open(image)
    ima = np.array(im) / 255.0

    prediction = model(np.array([ima]))
    prediction = tf.math.argmax(prediction, axis=-1)

    return "".join(map(chr, map(int, prediction[0])))


if __name__ == "__main__":
    # Check its architecture
    model.summary()

    print(predict("./captcha.jpeg"))
