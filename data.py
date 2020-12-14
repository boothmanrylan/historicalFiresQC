import itertools
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


# some earth engine bands come as tf.string and require tf.io.decode_raw to
# parse, others come as their expected dtype and dont, unclear why
string_bands = ['B4', 'B5', 'B6', 'B7', 'OldB4', 'OldB5',
                'OldB6', 'OldB7', 'class', 'noisyClass']
float_bands = ['dateDiff', 'referencePoints', 'mergedReferencePoints',
               'burnAge', 'mergedBurnAge']
int_bands = ['CART', 'stackedCART']

@tf.function
def parse(example, shape, image_bands, annotation_bands, combine=None):
    feature_description = {
        k: tf.io.FixedLenFeature((), tf.string) for k in string_bands
    }
    feature_description.update({
        k: tf.io.FixedLenFeature(shape, tf.float32) for k in float_bands
    })
    feature_description.update({
        k: tf.io.FixedLenFeature(shape, tf.int64) for k in int_bands
    })
    def stack_bands(parsed_example, band_names, output_dtype):
        return tf.cast(
            tf.stack([
                tf.reshape(tf.io.decode_raw(parsed_example[band], tf.uint8), shape)
                if band in string_bands
                else tf.reshape(parsed_example[band], shape)
                for band in band_names
            ], axis=-1),
            output_dtype
        )

    parsed = tf.io.parse_single_example(example, feature_description)

    annotation = stack_bands(parsed, annotation_bands, tf.int64)

    if combine is not None:
        for k, v in combine:
            annotation = tf.where(annotation == k, v, annotation)

    # the date difference needs to be scaled by a different value than the
    # other possible image bands therefore extract it on its own and concat
    # with the rest of image after scaling both separately
    if 'dateDiff' in image_bands:
        image_bands.remove('dateDiff')
        date_diff = tf.reshape(parsed.pop('dateDiff'), (*shape, 1))
        date_diff /= 1071 # hard coded max date difference
    else:
        date_diff = None

    image = stack_bands(parsed, image_bands, tf.float32)
    image /= 255.0

    if date_diff is not None:
        image = tf.concat([image, date_diff], -1)

    # stack_bands adds an extra dimension with shape 1 if called on a single
    # band, remove it here with squeeze (does nothing if no dim with shape 1
    # exists)
    return tf.squeeze(image), tf.squeeze(annotation)


@tf.function
def parse_all(example, shape, get_images=True, stack_image=False,
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
        k: tf.io.FixedLenFeature((), tf.string) for k in string_bands
    }
    feature_description.update({
        k: tf.io.FixedLenFeature(shape, tf.float32) for k in float_bands
    })
    feature_description.update({
        k: tf.io.FixedLenFeature(shape, tf.int64) for k in int_bands
    })

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

    date_difference = tf.reshape(parsed.pop('dateDiff'), (*shape, 1))

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

    image = tf.cast(
        tf.stack([
            tf.reshape(tf.io.decode_raw(parsed[k], tf.uint8), shape)
            for k in bands
        ], axis=-1),
        tf.float32
    )

    image /= 255.0

    if stack_image and include_date_difference:
        image = tf.concat([image, date_difference], -1)

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

def get_dataset(patterns, shape, image_bands, annotation_bands, combine=None,
                batch_size=64, filters=None, cache=False, shuffle=False,
                repeat=False, prefetch=False):

    if not isinstance(patterns, list):
        patterns = [patterns]
    if not isinstance(image_bands, list):
        image_bands = [image_bands]
    if not isinstance(annotation_bands, list):
        annotation_bands = [annotation_bands]

    files = tf.data.Dataset.list_files(patterns[0])

    for p in patterns[1:]:
        files = files.concatenate(tf.data.Dataset.list_files(p))

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(
        lambda x: parse(x, shape, image_bands, annotation_bands, combine),
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
