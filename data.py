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
        bands = [
            tf.cast(
                tf.reshape(
                    tf.io.decode_raw(parsed_example[band], tf.uint8),
                    shape
                ),
                output_dtype
            )
            if band in string_bands
            else tf.cast(parsed_example[band], output_dtype)
            for band in band_names
        ]
        return tf.stack(bands, axis=-1)

    parsed = tf.io.parse_single_example(example, feature_description)

    annotation = stack_bands(parsed, annotation_bands, tf.int64)

    if combine is not None:
        for original, change in combine:
            original = tf.cast(original, annotation.dtype)
            change = tf.cast(change, annotation.dtype)
            annotation = tf.where(annotation == original, change, annotation)

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
                batch_size=64, filters=True, cache=False, shuffle=False,
                repeat=False, prefetch=False):

    if not isinstance(patterns, list):
        patterns = [patterns]
    if not isinstance(image_bands, list):
        image_bands = [image_bands]
    if not isinstance(annotation_bands, list):
        annotation_bands = [annotation_bands]

    if '*' in patterns[0]: # patterns need unix style file expansion
        files = tf.data.Dataset.list_files(patterns[0])

        for p in patterns[1:]:
            files = files.concatenate(tf.data.Dataset.list_files(p))
    else: # pattern are complete file names
        files = tf.data.Dataset.list_files(patterns, shuffle=False)

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(
        lambda x: parse(x, shape, image_bands, annotation_bands, combine),
        num_parallel_calls=AUTOTUNE
    )

    if filters:
        dataset = dataset.filter(filter_blank).filter(filter_nan)
        if not isinstance(filters, bool):
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
