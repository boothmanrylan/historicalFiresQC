import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def parse(example, shape):
    feature_description = {
        'B4': tf.io.FixedLenFeature((), tf.string),
        'B5': tf.io.FixedLenFeature((), tf.string),
        'B6': tf.io.FixedLenFeature((), tf.string),
        'B7': tf.io.FixedLenFeature((), tf.string),
        'class': tf.io.FixedLenFeature((shape[0] * shape[1]), tf.int64)
    }

    parsed = tf.io.parse_single_example(example, feature_description)

    mask = tf.reshape(parsed.pop('class'), shape)

    image = tf.stack([
        tf.reshape(tf.io.decode_raw(parsed[k], tf.uint8), shape)
        for k in parsed.keys()
    ], axis=-1)
    image = tf.cast(image, tf.float32) / 255.0

    return image, mask

def filter_blank(image, mask):
    return not(tf.reduce_max(mask) == 0 and tf.reduce_min(mask) == 0)

def filter_no_burnt(image, mask):
    return tf.reduce_max(mask) == 4

def filter_nan(image, mask):
    return not tf.reduce_any(tf.math.is_nan(tf.cast(image, tf.float32)))

def get_dataset(patterns, shape, batch_size=64, filters=None, cache=True,
                shuffle=True, repeat=True, prefetch=True):
    try:
        len(patterns)
    except TypeError:
        patterns = [patterns]

    files = tf.data.Dataset.list_files(patterns[0])

    for p in patterns[1:]:
        files = files.concatenate(tf.data.Dataset.list_files(p))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(lambda x: parse(x, shape),
                          num_parallel_calls=AUTOTUNE)

    dataset = dataset.filter(filter_blank).filter(filter_nan)

    if filters is not None:
        try:
            len(filters)
        except TypeError:
            filters = [filters]
        for f in filters:
            dataset = dataset.filter(f)

    if cache:
        dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    if prefetch:
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
