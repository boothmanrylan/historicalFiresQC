import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def parse(example, shape, annotation=True, noisy_annotation=False,
          combine_burnt=False):

    if not (annotation or noisy_annotation):
        annotation = True

    feature_description = {
        'B4':         tf.io.FixedLenFeature((), tf.string),
        'B5':         tf.io.FixedLenFeature((), tf.string),
        'B6':         tf.io.FixedLenFeature((), tf.string),
        'B7':         tf.io.FixedLenFeature((), tf.string),
        'class':      tf.io.FixedLenFeature((), tf.string),
        'noisyClass': tf.io.FixedLenFeature((), tf.string)
    }

    parsed = tf.io.parse_single_example(example, feature_description)

    annotation = tf.reshape(
        tf.io.decode_raw(parsed.pop('class'), tf.uint8),
        shape
    )

    noisy_annotation = tf.reshape(
        tf.io.decode_raw(parsed.pop('noisyClass'), tf.uint8),
        shape
    )

    if combine_burnt:
        drop = tf.cast(5, annotation.dtype)
        replace = tf.cast(4, annotation.dtype)
        annotation = tf.where(annotation == drop, replace, annotation)
        noisy_annotation = tf.where(annotation == drop, replace, anotation)

    image = tf.cast(
        tf.stack([
            tf.reshape(tf.io.decode_raw(parsed[k], tf.uint8), shape)
            for k in parsed.keys()
        ], axis=-1),
        tf.float32
    )

    output = [image / 255.0]
    if annotation:
        output.append(annotation)
    if noisy_annotation:
        output.append(noisy_annotation)

    return output

def filter_blank(image, annotation, noisy_annotation=None):
    return (tf.reduce_min(annotation) != 0 or
            tf.reduce_max(annotation != 0))

def filter_no_x(x, image, annotation, noisy_annotation=None):
    compare = tf.cast(tf.fill(tf.shape(annotation), x), annotation.dtype)
    return tf.reduce_any(tf.equal(annotation, compare))

def filter_no_burnt(image, annotation, noisy_annotation=None):
    return (filter_no_x(4, image, annotation, noisy_annotation) or
            filter_no_x(5, image, annotation, noisy_annotation))

def filter_nan(image, annotation, noisy_annotation=None):
    return not tf.reduce_any(tf.math.is_nan(tf.cast(image, tf.float32)))

def get_dataset(patterns, shape, batch_size=64, filters=None, cache=True,
                shuffle=True, repeat=True, prefetch=True,
                annotation=True, noisy_annotation=False,
                combine_burnt=False):
    if not isinstance(patterns, list):
        patterns = [patterns]

    files = tf.data.Dataset.list_files(patterns[0])

    for p in patterns[1:]:
        files = files.concatenate(tf.data.Dataset.list_files(p))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(
        lambda x: parse(x, shape, annotation, noisy_annotation, combine_burnt),
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.filter(filter_blank).filter(filter_nan)

    if filters is not None:
        if not isinstance(filters, list):
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
