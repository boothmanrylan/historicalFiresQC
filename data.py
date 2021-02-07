import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

all_bands = [
    'B4', 'B5', 'B6', 'B7', 'OldB4', 'OldB5', 'OldB6', 'OldB7',
    'dateDiff', 'prevLSliceBurnAge', 'prevBBoxBurnAge', 'lsliceClass',
    'bboxClass', 'lsliceBurnAge', 'bboxBurnAge', 'lsliceBurnEdges',
    'bboxBurnEdges', 'referencePoints', 'refBurnAge'
]


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


def scale_burn_age(burn_age):
    return burn_age / (3650)


def inverse_scale_burn_age(burn_age):
    return burn_age * 3650


def log_burn_age(burn_age):
    return tf.math.log(burn_age + 1)


def inverse_log_burn_age(burn_age):
    return tf.math.exp(burn_age) - 1


def sigmoid_burn_age(burn_age):
    return 2 * tf.math.sigmoid(burn_age / 2) - 1


def inverse_sigmoid_burn_age(burn_age):
    return -2 * tf.math.log((2 / (burn_age + 1)) - 1)


@tf.function
def _add_separately(band_name, image_bands, parsed_example, shape, scale_fn):
    '''Helper function for parse.'''
    if band_name in image_bands:
        image_bands.remove(band_name)
        band = scale_fn(tf.reshape(parsed_example.pop(band_name), (*shape, 1)))
    else:
        band = None
    return band


@tf.function
def _stack_bands(parsed_example, band_names, dtype, shape):
    '''Helper function for parse.'''
    if band_names:
        bands = [tf.cast(parsed_example[b], dtype) for b in band_names]
        output = tf.squeeze(tf.stack(bands, axis=-1))
    else:
        output = tf.experimental.numpy.empty(shape, dtype=dtype)
    return output


@tf.function
def parse(example, shape, image_bands, annotation_bands, extra_bands=None,
          combine=None, burn_age_function=None):
    # make copies of input lists to avoid tensorflow could not parse source
    # code no matching AST error
    image_bands = image_bands.copy()
    annotation_bands = annotation_bands.copy()
    if extra_bands is not None:
        extra_bands = extra_bands.copy()

    feature_description = {
        k: tf.io.FixedLenFeature(shape, tf.float32) for k in all_bands
    }

    if burn_age_function is None:
        burn_age_function = scale_burn_age

    parsed = tf.io.parse_single_example(example, feature_description)

    lslice_burn_age = _add_separately(
        'lsliceBurnAge', annotation_bands, parsed, shape, burn_age_function
    )

    bbox_burn_age = _add_separately(
        'bboxBurnAge', annotation_bands, parsed, shape, burn_age_function
    )

    if lslice_burn_age is not None or bbox_burn_age is not None:
        annotation_dtype = tf.float32
    else:
        annotation_dtype = tf.int64

    annotation = _stack_bands(parsed, annotation_bands, annotation_dtype, shape)

    if combine is not None:
        for original, change in combine:
            original = tf.cast(original, annotation.dtype)
            change = tf.cast(change, annotation.dtype)
            annotation = tf.where(annotation == original, change, annotation)

    if lslice_burn_age is not None:
        annotation = tf.concat([annotation, tf.squeeze(lslice_burn_age)], -1)

    if bbox_burn_age is not None:
        annotation = tf.concat([annotation, tf.squeeze(bbox_burn_age)], -1)

    date_diff = _add_separately(
        'dateDiff', image_bands, parsed, shape, lambda x: x / 1071
    )

    prev_bbox_burn_age = _add_separately(
        'prevBBoxBurnAge', image_bands, parsed, shape, burn_age_function
    )

    prev_lslice_burn_age = _add_separately(
        'prevLSliceBurnAge', image_bands, parsed, shape, burn_age_function
    )

    image = _stack_bands(parsed, image_bands, tf.float32, shape)
    image /= 255.0 # MSS data is unsigned 8 bit integer therefore 255 is max

    if date_diff is not None:
        image = tf.concat([image, date_diff], -1)

    if prev_bbox_burn_age is not None:
        image = tf.concat([image, prev_bbox_burn_age], -1)

    if prev_lslice_burn_age is not None:
        image = tf.concat([image, prev_lslice_burn_age], -1)

    if extra_bands is not None:
        extra = _stack_bands(parsed, extra_bands, tf.float32, shape)
        return image, annotation, extra

    return image, annotation


def get_dataset(patterns, shape, image_bands, annotation_bands,
                extra_bands=None, combine=None, batch_size=64, filters=True,
                cache=False, shuffle=False, repeat=False, prefetch=False,
                burn_age_function=None):
    """
    Create a TFRecord dataset.

    patterns (str/str list): Files to create dataset from. Can either be
        complete filepaths or unix style patterns.
    shape (int tuple): Shape of each patch without band dimension.
    image_bands (str list): Name of each band to use in model input.
    annotation_bands (str list): Name of each band to use in ground truth.
    extra_bands (str list): Name of bands to return as an addition image e.g.
        to mask out the edges of burns in the loss calculation.
    combine ((int, int) list): For each tuple in combine replace all pixels in
        annotation that eqaul the first value with the second value.
    filters (bool or function list): If false no filters are applied, if true
        any patches with NaN or whose annotations are all blank are filtered
        out, if a function list in addition to the NaN and blank filter the
        functions are also applied as filters.
    cache (bool): if true cache the dataset
    shuffle (bool): if true shuffle dataset with buffer size 1000.
    repeat (bool): if true dataset repeates infinitely.
    prefetch (bool): if true the dataset is prefetched with AUTOTUNE buffer.
    burn_age_function (function): If given, applied to burn age during parse.

    Returns a tf.data.TFRecordDataset
    """

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

    # ensure band names are all valid
    for b in image_bands:
        try:
            assert b in all_bands
        except AssertionError as E:
            raise ValueError(f'invalid image band name: {b}') from E
    for b in annotation_bands:
        try:
            assert b in all_bands
        except AssertionError as E:
            raise ValueError(f'invalid annotation band name: {b}') from E

    # set combine to None if predicting burn age
    if ('lsliceBurnAge' in annotation_bands or
        'bboxBurnAge' in annotation_bands):
        combine=None

    dataset = tf.data.TFRecordDataset(files, compression_type='GZIP')
    dataset = dataset.map(
        lambda x: parse(x, shape, image_bands, annotation_bands,
                        extra_bands, combine, burn_age_function),
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
