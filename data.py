import itertools
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

@tf.function
def parse(example, shape, get_images=True, stack_image=False,
          include_date_difference=False, clean_annotation=False,
          noisy_annotation=False, combined_burnt=False, split_burnt=False,
          get_ref_points=False, get_merged_ref_points=False,
          get_burn_age=False, get_merged_burn_age=False,
          get_CART_classification=False,
          get_stacked_CART_classification=False):
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

    feature_description = {
        'B4':         tf.io.FixedLenFeature((), tf.string),
        'B5':         tf.io.FixedLenFeature((), tf.string),
        'B6':         tf.io.FixedLenFeature((), tf.string),
        'B7':         tf.io.FixedLenFeature((), tf.string),
        'OldB4':      tf.io.FixedLenFeature((), tf.string),
        'OldB5':      tf.io.FixedLenFeature((), tf.string),
        'OldB6':      tf.io.FixedLenFeature((), tf.string),
        'OldB7':      tf.io.FixedLenFeature((), tf.string),
        'dateDiff':   tf.io.FixedLenFeature(shape, tf.float32),
        'class':      tf.io.FixedLenFeature((), tf.string),
        'noisyClass': tf.io.FixedLenFeature((), tf.string),
        'CART':       tf.io.FixedLenFeature(shape, tf.int64),
        'stackedCART': tf.io.FixedLenFeature(shape, tf.int64),
        'referencePoints': tf.io.FixedLenFeature(shape, tf.float32),
        'mergedReferencePoints': tf.io.FixedLenFeature(shape, tf.float32),
        'burnAge': tf.io.FixedLenFeature(shape, tf.float32),
        'mergedBurnAge': tf.io.FixedLenFeature(shape, tf.float32)
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

    ref_points = tf.reshape(parsed.pop('referencePoints'), shape)

    merged_ref_points = tf.reshape(parsed.pop('mergedReferencePoints'), shape)

    burn_age = tf.reshape(parsed.pop('burnAge'), shape)

    merged_burn_age = tf.reshape(parsed.pop('mergedBurnAge'), shape)

    CART_classification = tf.reshape(parsed.pop('CART'), shape)

    stacked_CART_classification = tf.reshape(parsed.pop('stackedCART'), shape)

    date_difference = tf.reshape(parsed.pop('dateDiff'), shape)

    date_difference /= 1071 # hard coded maximum date difference

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

    bands = ['B4', 'B5', 'B6', 'B7']
    if stack_image:
        bands.extend(['OldB4', 'OldB5', 'OldB6', 'OldB7'])
        if include_date_difference:
            bands.append('dateDiff')

    image = tf.cast(
        tf.stack([
            tf.reshape(tf.io.decode_raw(parsed[k], tf.uint8), shape)
            for k in bands
        ], axis=-1),
        tf.float32
    )

    image /= 255.0

    outputs = [image,
               combined_burnt_clean_annotation,
               split_burnt_clean_annotation,
               combined_burnt_noisy_annotation,
               split_burnt_noisy_annotation,
               CART_classification,
               stacked_CART_classification,
               ref_points,
               merged_ref_points,
               burn_age,
               merged_burn_age]

    selectors = [get_images,
                 combined_burnt and clean_annotation,
                 split_burnt and clean_annotation,
                 combined_burnt and noisy_annotation,
                 split_burnt and noisy_annotation,
                 get_CART_classification,
                 get_stacked_CART_classification,
                 get_ref_points,
                 get_merged_ref_points,
                 get_burn_age,
                 get_merged_burn_age]

    return list(itertools.compress(outputs, selectors))

def filter_blank(image, annotation, *annotations):
    return not (tf.reduce_min(annotation) == 0 and
                tf.reduce_max(annotation) == 0)

def filter_no_x(x, image, annotation, *annotations):
    compare = tf.cast(tf.fill(tf.shape(annotation), x), annotation.dtype)
    return tf.reduce_any(tf.equal(annotation, compare))

def filter_no_burnt(image, annotation, *annotations):
    return (filter_no_x(4, image, annotation) or
            filter_no_x(5, image, annotation))

def filter_nan(image, annotation, *annotations):
    return not tf.reduce_any(tf.math.is_nan(tf.cast(image, tf.float32)))

def get_dataset(patterns, shape, batch_size=64, filters=None, cache=False,
                shuffle=False, repeat=False, prefetch=False, get_images=True,
                stack_image=False, include_date_difference=False,
                clean_annotation=False, noisy_annotation=False,
                combined_burnt=False, split_burnt=False, get_ref_points=False,
                get_merged_ref_points=False, get_burn_age=False,
                get_merged_burn_age=False, get_CART_classification=False,
                get_stacked_CART_classification=False):
    if not isinstance(patterns, list):
        patterns = [patterns]

    files = tf.data.Dataset.list_files(patterns[0])

    for p in patterns[1:]:
        files = files.concatenate(tf.data.Dataset.list_files(p))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(
        lambda x: parse(x, shape, get_images, stack_image,
                        include_date_difference, clean_annotation,
                        noisy_annotation, combined_burnt, split_burnt,
                        get_ref_points, get_merged_ref_points, get_burn_age,
                        get_merged_burn_age, get_CART_classification,
                        get_stacked_CART_classification),
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
