import tensorflow as tf

feature_description = {
    'B4': tf.io.FixedLenFeature((), tf.string),
    'B5': tf.io.FixedLenFeature((), tf.string),
    'B6': tf.io.FixedLenFeature((), tf.string),
    'B7': tf.io.FixedLenFeature((), tf.string),
    'class': tf.io.FixedLenFeature((128 * 128,), tf.int64)
}
