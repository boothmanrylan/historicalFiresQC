import tensorflow as tf
import itertools

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def parse(example, shape, clean_annotation=True, noisy_annotation=False,
          combined_burnt=True, split_burnt=False):
    """
    Parse TFRecords containing annotated MSS data.

    Assumes TFRecords contain both a "clean" and a "noisy" annotation. Either
    or both of the annotations can be returned by setting clean_annotation
    and/or noisy_annotation to True. Defaults to just return the clean
    annotations.

    Assumes the numerically largest and second largest classes represent new
    and old burns. The burn classes can be returned as separate classes and/or
    as one class by setting combined_burnt and/or split_burnt to True. Defaults
    to returning them as one class.

    Regardless of how many or which combinations of clean/noisy combined/split
    are returned the output will always be in this order:
    image, combined clean, split clean, combined clean, split noisy
    """

    # ensure at least one of clean_annotation or noisy_annotation is True
    if not (clean_annotation or noisy_annotation):
        clean_annotation = True

    # ensure at least one of combined_burnt or split_burnt is True
    if not (combined_burnt or split_burnt):
        combined_burnt = True

    feature_description = {
        'B4':         tf.io.FixedLenFeature((), tf.string),
        'B5':         tf.io.FixedLenFeature((), tf.string),
        'B6':         tf.io.FixedLenFeature((), tf.string),
        'B7':         tf.io.FixedLenFeature((), tf.string),
        'class':      tf.io.FixedLenFeature((), tf.string),
        'noisyClass': tf.io.FixedLenFeature((), tf.string)
    }

    parsed = tf.io.parse_single_example(example, feature_description)

    split_burnt_clean_annotation = tf.reshape(
        tf.io.decode_raw(parsed.pop('class'), tf.uint8),
        shape
    )

    split_burnt_noisy_annotation = tf.reshape(
        tf.io.decode_raw(parsed.pop('noisyClass'), tf.uint8),
        shape
    )

    drop =  tf.reduce_max(split_burnt_clean_annotation)
    replace =  drop - 1
    combined_burnt_clean_annotation = tf.where(
        split_burnt_clean_annotation == drop,
        replace,
        split_burnt_clean_annotation)
    combined_burnt_noisy_annotation = tf.where(
        split_burnt_noisy_annotation == drop,
        replace,
        split_burnt_noisy_annotation
    )

    image = tf.cast(
        tf.stack([
            tf.reshape(tf.io.decode_raw(parsed[k], tf.uint8), shape)
            for k in parsed.keys()
        ], axis=-1),
        tf.float32
    )

    outputs = [image / 255.0,
               combined_burnt_clean_annotation,
               split_burnt_clean_annotation,
               combined_burnt_noisy_annotation,
               split_burnt_noisy_annotation]

    selectors = [True,
                 combined_burnt and clean_annotation,
                 split_burnt and clean_annotation,
                 combined_burnt and noisy_annotation,
                 split_burnt and noisy_annotation]

    return list(itertools.compress(outputs, selectors))

def filter_blank(image, annotation, *annotations):
    return not (tf.reduce_min(annotation) == 0 and
                tf.reduce_max(annotation) == 0)

def filter_no_x(x, image, annotation, *annotations):
    compare = tf.cast(tf.fill(tf.shape(annotation), x), annotation.dtype)
    return tf.reduce_any(tf.equal(annotation, compare))

def filter_no_burnt(image, annotation, *annotations):
    return (filter_no_x(4, image, annotation, noisy_annotation) or
            filter_no_x(5, image, annotation, noisy_annotation))

def filter_nan(image, annotation, *annotations):
    return not tf.reduce_any(tf.math.is_nan(tf.cast(image, tf.float32)))

def get_dataset(patterns, shape, batch_size=64, filters=None, cache=True,
                shuffle=True, repeat=True, prefetch=True,
                clean_annotation=True, noisy_annotation=False,
                combined_burnt=True, split_burnt=False):
    if not isinstance(patterns, list):
        patterns = [patterns]

    files = tf.data.Dataset.list_files(patterns[0])

    for p in patterns[1:]:
        files = files.concatenate(tf.data.Dataset.list_files(p))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(
        lambda x: parse(x, shape, clean_annotation, noisy_annotation,
                        combined_burnt, split_burnt),
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
