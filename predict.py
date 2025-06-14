import tensorflow as tf

from utils import inverse_string_lookup


def predict(model, img):
    input_data = tf.io.decode_jpeg(img)
    input_data = tf.expand_dims(input_data, axis=0)
    input_data = tf.cast(input_data, dtype=tf.float32)

    # Make predictions
    predictions = model.call(input_data)
    predictions = b"".join(inverse_string_lookup(predictions[-1][0]).numpy()).decode()

    return predictions


if __name__ == "__main__":
    import sys

    model = tf.keras.layers.TFSMLayer(sys.argv[1])

    with open(sys.argv[2], "rb") as f:
        img = f.read()

    # Get predictions
    output = predict(model, img)

    print("Predicted output:", output)
