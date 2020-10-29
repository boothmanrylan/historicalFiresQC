import tensorflow as tf
from utils import feature_description

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def parse(example):
    parsed = tf.io.parse_single_example(example, feature_description)
    mask = tf.reshape(parsed.pop('class'), SHAPE)
    image = tf.stack([
        tf.reshape(tf.io.decode_raw(parsed[k], tf.uint8), SHAPE)
        for k in parsed.keys()
    ], axis=-1)
    image = tf.cast(image, tf.float32) / 255.0
    return image, mask

def filter_blank(image, mask):
    return not(tf.reduce_max(mask) == 0 and tf.reduce_min(mask) == 0)

def filter_no_burnt(image, maks):
    return tf.reduce_max(mask) == 4

def filter_nan(image, mask):
    return not tf.reduce_any(tf.math.is_nan(tf.cast(image, tf.float32)))

def get_dataset(patterns, batch_size=64, filters=None, cache=True,
             shuffle=True, repeat=True, prefetch=True):
    try:
        len(patterns)
    except TypeError:
        patterns = [patterns]

    files = tf.data.Dataset.list_file(patterns[0])

    for p in patterns[1:]:
        files = files.concatenate(tf.data.Dataset.list_files(p))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(parse, num_parallel_calls=AUTOTUNE)

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
