import tensorflow as tf

from config import CASE_SENSITIVE

CHARSET = "abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

normalize_label = tf.identity
if not CASE_SENSITIVE:
    CHARSET = CHARSET.upper()
    normalize_label = tf.strings.upper
vocabulary = sorted(set(CHARSET))

# Mapping characters to integers
string_lookup = tf.keras.layers.StringLookup(
    vocabulary=vocabulary, mask_token=None, num_oov_indices=0
)
inverse_string_lookup = tf.keras.layers.StringLookup(
    vocabulary=vocabulary, mask_token=None, num_oov_indices=0, invert=True
)
