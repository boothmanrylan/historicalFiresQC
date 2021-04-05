import math
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE

all_bands = [
    'B4', 'B5', 'B6', 'B7', 'OldB4', 'OldB5', 'OldB6', 'OldB7',
    'dateDiff', 'prevLSliceBurnAge', 'prevBBoxBurnAge', 'lsliceClass',
    'bboxClass', 'lsliceBurnAge', 'bboxBurnAge', 'lsliceBurnEdges',
    'bboxBurnEdges', 'referencePoints', 'refBurnAge', 'prevLSliceClass',
    'prevBBoxClass', 'TCA', 'OldTCA', 'Brightness', 'Greenness', 'Yellowness',
    'Nonesuch', 'OldBrightness', 'OldGreeness', 'OldYellowness', 'OldNonesuch'
]

def filter_blank(image, _):
    return not tf.reduce_max(image) == 0


def filter_no_x(x, _, annotation):
    if tf.shape(annotation).shape[0] > 2:
        annot = annotation[:, :, 0]
        compare = tf.cast(tf.fill(tf.shape(annot), x), annot.dtype)
        output = tf.reduce_any(tf.equal(annot, compare))
    else:
        compare = tf.cast(tf.fill(tf.shape(annotation), x), annotation.dtype)
        output = tf.reduce_any(tf.equal(annotation, compare))
    return output


def filter_x(x, image, annotation):
    return not filter_no_x(x, image, annotation)


def filter_mostly_burnt(_, annotation, burn=3, percent=0.5):
    if tf.shape(annotation).shape[0] > 2:
        annot = annotation[:, :, 0]
    else:
        annot = annotation
    compare = tf.cast(tf.fill(tf.shape(annot), burn), annot.dtype)
    numerator = tf.reduce_sum(tf.cast(tf.equal(annot, compare), tf.int64))
    denominator = tf.cast(tf.reduce_prod(tf.shape(annot)), numerator.dtype)
    return numerator / denominator > percent


def filter_all_max_burn_age(_, annotation):
    if tf.shape(annotation).shape[0] > 2:
        # annotation has extra bands in it e.g. to mask burn edges
        # dont use the extra bands
        annot = annotation[:, :, 0]
        output = not tf.reduce_all(annot == tf.reduce_max(annot))
    else:
        output = not tf.reduce_all(annotation == tf.reduce_max(annotation))
    return output


def filter_no_burnt(image, annotation):
    return (filter_no_x(4, image, annotation) or
            filter_no_x(5, image, annotation))


def filter_nan(image, _):
    return not tf.reduce_any(tf.math.is_nan(tf.cast(image, tf.float32)))

def scale_burn_age(burn_age):
    return burn_age / (3650)


def inverse_scale_burn_age(burn_age):
    return burn_age * 3650


def log_burn_age(burn_age):
    return tf.math.log((burn_age / 365) + 1)


def inverse_log_burn_age(burn_age):
    return 365 * (tf.math.exp(burn_age) - 1)


def sigmoid_burn_age(burn_age):
    return 2 * tf.math.sigmoid(burn_age / 730) - 1


def inverse_sigmoid_burn_age(burn_age):
    return -730 * tf.math.log((2 / (burn_age + 1)) - 1)


def _add_separately(band_name, image_bands, parsed_example, shape, scale_fn):
    '''Helper function for parse.'''
    if band_name in image_bands:
        band = scale_fn(tf.reshape(parsed_example[band_name], (*shape, 1)))
    else:
        band = None
    return band


def _stack_bands(parsed_example, band_names, dtype):
    '''Helper function for parse.'''
    if band_names:
        bands = [tf.cast(parsed_example[b], dtype) for b in band_names]
        output = tf.squeeze(tf.stack(bands, axis=-1))
    else:
        output = None
    return output


@tf.function
def parse(example, shape, image_bands, annotation_bands, combine=None,
          burn_age_function=None, default_scale_fn=lambda x: x / 255):

    used_bands = image_bands + annotation_bands

    feature_description = {
        k: tf.io.FixedLenFeature(shape, tf.float32) for k in used_bands
    }

    if burn_age_function is None:
        burn_age_function = scale_burn_age

    parsed = tf.io.parse_single_example(example, feature_description)

    # burn age bands must be scaled, but regular annotation bands should not be
    # scaled therefore add them each separately to the annotation
    lslice_burn_age = _add_separately(
        'lsliceBurnAge', annotation_bands, parsed, shape, burn_age_function
    )

    bbox_burn_age = _add_separately(
        'bboxBurnAge', annotation_bands, parsed, shape, burn_age_function
    )

    # ensure burn age bands are not added to the annotation twice
    new_annotation_bands = annotation_bands.copy()
    for band in ['lsliceBurnAge', 'bboxBurnAge']:
        if band in annotation_bands:
            new_annotation_bands.remove(band)

    if lslice_burn_age is not None or bbox_burn_age is not None:
        annot_dtype = tf.float32
    else:
        annot_dtype = tf.int64

    annotation = _stack_bands(parsed, new_annotation_bands, annot_dtype)

    if combine is not None: # combine should only be applied to the first band
        for original, change in combine:
            original = tf.cast(original, annotation.dtype)
            change = tf.cast(change, annotation.dtype)
            if len(tf.shape(annotation)) > 2:
                bands = tf.split(annotation, [1, annotation.shape[-1] - 1], -1)
                combined = tf.where(bands[0] == original, change, bands[0])
                annotation = tf.concat([combined, bands[1]], -1)
            else:
                annotation = tf.where(annotation == original, change, annotation)

    if lslice_burn_age is not None:
        if annotation is not None:
            annotation = tf.concat([annotation, lslice_burn_age], -1)
        else:
            annotation = tf.squeeze(lslice_burn_age)

    if bbox_burn_age is not None:
        if annotation is not None:
            annotation = tf.concat([annotation, bbox_burn_age], -1)
        else:
            annotation = tf.squeeze(bbox_burn_age)

    # some bands must be scaled differently than MSS band add them separately
    separate_band_names = {
        'dateDiff': lambda x: x / 1071,
        'prevBBoxBurnAge': burn_age_function,
        'prevLSliceBurnAge': burn_age_function,
        'prevLSliceClass': lambda x: x,
        'prevBBoxClass': lambda x: x
    }

    separate_bands = [
        _add_separately(band_name, image_bands, parsed, shape, scale_fn)
        for band_name, scale_fn in separate_band_names.items()
    ]

    # ensure the bands added separately are not added twice
    new_image_bands = image_bands.copy()
    for band in separate_band_names:
        if band in image_bands:
            new_image_bands.remove(band)

    image = _stack_bands(parsed, new_image_bands, tf.float32)
    image = default_scale_fn(image)

    for band in separate_bands:
        if band is not None:
            if image is not None:
                image = tf.concat([image, band], -1)
            else:
                image = tf.squeeze(band)

    return image, annotation

# used by augment data to randomly flip, rotate and zoom training data
augmenter = tf.keras.Sequential([
    preprocessing.RandomFlip("horizontal_and_vertical"),
    preprocessing.RandomRotation(0.3),
    preprocessing.RandomZoom(0.2)
])

@tf.function
def augment_data(x, y):
    """
    randomly flips, rotates, and zooms x and y in identical fashion
    randomly adjusts the brightness and contrast of x
    """
    # save shapes to explicitly set the shape of the outputs
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)

    if len(y_shape) == 3: # add dummy channel dimension, removed later with squeeze
        y = tf.reshape(y, tf.concat([y_shape, [1]], -1))
        n_y_bands = 1
    else:
        n_y_bands = y_shape[-1]

    # cast y to same type as x before concat, undone before returning
    y_type = y.dtype
    y = tf.cast(y, x.dtype)

    # stack x and y before augmenting so that pixels in both have their
    # locations changed in an indentical manner
    xy = augmenter(tf.concat([x, y], -1), training=True)

    # split the combined x and y
    new_x = xy[:, :, :, :-n_y_bands]
    new_y = tf.cast(tf.squeeze(xy[:, :, :, -n_y_bands:]), y_type)

    g = tf.random.get_global_generator()
    seed = g.uniform_full_int((2,), tf.dtypes.int32)

    # randomly adjust the brightness and contrast of x
    if g.normal((1,))[0] < 0.67: # ~75% chance this happens
        new_x = tf.image.stateless_random_contrast(new_x, 0.1, 0.5, seed=seed)
    if g.normal((1,))[0] < 0.67:
        new_x = tf.image.stateless_random_brightness(new_x, 0.5, seed=seed)

    # add random gaussian noise to x
    if g.normal((1,))[0] < 0.67:
        add_noise = tf.cast(g.normal(tf.shape(new_x)), new_x.dtype)
        # scale before adding because x values always in range 0, 1
        new_x = new_x + (add_noise / tf.cast(100.0, new_x.dtype))

    # explicit reshape to avoid
    # ValueError: as_list() is not defined on an unknown TensorShape.
    # which is thrown by model.fit
    new_x = tf.reshape(x, x_shape)
    new_y = tf.reshape(y, y_shape)
    return  new_x, new_y


def get_dataset(patterns, shape, image_bands, annotation_bands,
                combine=None, batch_size=64, filters=True, cache=False,
                shuffle=False, repeat=False, prefetch=False,
                burn_age_function=None, augment=False, percent_burn_free=None,
                burn_class=2):
    """
    Create a TFRecord dataset.

    patterns (str/str list): Files to create dataset from. Can either be
        complete filepaths or unix style patterns.
    shape (int tuple): Shape of each patch without band dimension.
    image_bands (str list): Name of each band to use in model input.
    annotation_bands (str list): Name of each band to use in ground truth.
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
    augment (bool): If true the data is augmented with random flips etc...
    burn_class (int): The value that represents a burnt pixel.
    percent_burn_free (None or float): If float, the percent of patches that
        contain zero burned pixels

    Returns a tf.data.TFRecordDataset
    """

    if not isinstance(patterns, list):
        patterns = [patterns]
    if not isinstance(image_bands, list):
        image_bands = [image_bands]
    if not isinstance(annotation_bands, list):
        annotation_bands = [annotation_bands]

    default_scale_fn = lambda x: x / 255
    if 'NormalizedData' in patterns[0] or 'MaskedData' in patterns[0]:
        # normalized/masked data is already scaled, therefore do nothing
        default_scale_fn = lambda x: x

    if '*' in patterns[0]: # patterns need unix style file expansion
        files = tf.data.Dataset.list_files(patterns[0], shuffle=shuffle)
        for p in patterns[1:]:
            files = files.concatenate(
                tf.data.Dataset.list_files(p, shuffle=shuffle)
            )
    else: # pattern are complete file names
        files = tf.data.Dataset.list_files(patterns, shuffle=shuffle)

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
        lambda x: parse(x, shape, image_bands, annotation_bands, combine,
                        burn_age_function, default_scale_fn),
        num_parallel_calls=AUTOTUNE
    )

    if percent_burn_free is not None:
        burn_free_dataset = dataset.filter(filter_blank).filter(filter_nan)
        burn_free_dataset = burn_free_dataset.filter(
            lambda im, annot: filter_x(burn_class, im, annot)
        )
        if shuffle:
            burn_free_dataset = burn_free_dataset.shuffle(1000)

    if filters:
        dataset = dataset.filter(filter_blank).filter(filter_nan)
        if not isinstance(filters, bool):
            if not isinstance(filters, list):
                filters = [filters]
            for f in filters:
                dataset = dataset.filter(f)

    if percent_burn_free is not None:
        # based on https://stackoverflow.com/a/58573644
        burn_free_ratio = int(percent_burn_free * 100)
        burn_ratio = 100 - burn_free_ratio
        gcd = math.gcd(burn_free_ratio, burn_ratio)
        d1 = burn_free_dataset.batch(burn_free_ratio // gcd)
        d2 = dataset.batch(burn_ratio // gcd)
        combined = tf.data.Dataset.zip((d1, d2)).map(
            lambda a, b:
                (tf.concat((a[0], b[0]), 0), tf.concat((a[1], b[1]), 0))
        )
        dataset = combined.unbatch()

    if cache:
        dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)

    if augment:
        dataset = dataset.map(augment_data)

    if repeat:
        dataset = dataset.repeat()

    if prefetch:
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
