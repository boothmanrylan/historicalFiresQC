import tensorflow as tF

feature_description = {
    'B4': tf.io.FixedLenFeature((), tf.string),
    'B5': tf.io.FixedLenFeature((), tf.string),
    'B6': tf.io.FixedLenFeature((), tf.string),
    'B7': tf.io.FixedLenFeature((), tf.string),
    'class': tf.io.FixedLenFeature((SHAPE[0] * SHAPE[1],), tf.int64)
}
