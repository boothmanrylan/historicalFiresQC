import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

AUTOTUNE = tf.data.experimental.AUTOTUNE
RNG = tf.random.Generator.from_seed(123, alg='philox')
ALL_BANDS = ['nir', 'red_edge', 'red', 'green', 'bai', 'tca', 'class']
AUGMENTER = tf.keras.Sequential([
    preprocessing.RandomFlip("horizontal_and_vertical"),
    preprocessing.RandomRotation(0.3),
    preprocessing.RandomZoom(0.2)
])


@tf.function
def parse(example, shape, image_bands, annotation_bands):
    """
    Converts TFRecord example into Tensor ready to be passed to a model.

    returns a tensor tuple: image, annotation
    """
    if annotation_bands is not None:
        used_bands = image_bands + annotation_bands
    else:
        used_bands = image_bands

    feature_description = {
        k: tf.io.FixedLenFeature(shape, tf.float32) for k in used_bands
    }

    parsed = tf.io.parse_single_example(example, feature_description)

    image = tf.stack(
        [tf.reshape(parsed[x], shape) for x in image_bands], -1
    )
    if annotation_bands is not None:
        annotation = tf.cast(tf.stack(
            [tf.reshape(parsed[x], shape) for x in annotation_bands], -1
        ), tf.int64)
        annotation = tf.squeeze(annotation)
    else:
        annotation = None

    return tf.squeeze(image), annotation


@tf.function
def _augment_data(x, y, seed):
    """
    Helper function for augment_data. Necessary because we want each call to
    _augment_data to have a new seed.

    Randomly flips, rotates, and zooms x and y in identical fashion
    Randomly adjusts the brightness and contrast of x

    Returns x, y after applying the augmentations.
    """
    # save shapes to explicitly set the shape of the outputs
    x_shape = tf.shape(x)
    y_shape = tf.shape(y)

    if len(y_shape) == 2:  # add dummy channel dimension
        y = tf.reshape(y, tf.concat([y_shape, [1]], -1))
        n_y_bands = 1
    else:
        n_y_bands = y_shape[-1]

    # cast y to same type as x before concat, undone before returning
    y_type = y.dtype
    y = tf.cast(y, x.dtype)

    # stack x and y before augmenting so that pixels in both have their
    # locations changed in an indentical manner
    xy = AUGMENTER(tf.concat([x, y], -1), training=True)

    # split the combined x and y
    new_x = xy[:, :, :-n_y_bands]
    new_y = tf.cast(tf.squeeze(xy[:, :, -n_y_bands:]), y_type)

    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    # randomly adjust the brightness and contrast of x
    new_x = tf.image.stateless_random_contrast(new_x, 0.1, 0.5, seed=new_seed)
    new_x = tf.image.stateless_random_brightness(new_x, 0.5, seed=new_seed)

    # add random gaussian noise to x
    add_noise = tf.cast(RNG.normal(tf.shape(new_x)), new_x.dtype)
    # scale before adding because x values always in range 0, 1
    new_x += (add_noise / tf.cast(100.0, new_x.dtype))

    # explicit reshape to avoid
    # ValueError: as_list() is not defined on an unknown TensorShape.
    # which is thrown by model.fit
    new_x = tf.reshape(new_x, x_shape)
    new_y = tf.reshape(new_y, y_shape)
    return new_x, new_y


def augment_data(x, y):
    """
    Creates random seed then calls _augment_data to perform data augmentation

    x (tensor): image
    y (tensor): image annotation

    returns x, y after applying augmentation
    """
    seed = RNG.make_seeds(2)[0]
    return _augment_data(x, y, seed)


@tf.function
def _crop_data(x, y, height, width, seed):
    """
    Randomly crops x and y to have shape (height, width, bands).
    Ensures that both x and y are cropped in the same place.
    """
    # add dummy channel dimension to y if it has none
    y_shape = tf.shape(y)
    if len(y_shape) == 2:
        y = tf.reshape(y, tf.concat([y_shape, [1]], -1))
        n_y_bands = 1
    else:
        n_y_bands = y_shape[-1]

    n_x_bands = tf.shape(x)[-1]

    # cast y to same type as x before concat, undone before returning
    y_type = y.dtype
    y = tf.cast(y, x.dtype)

    xy = tf.concat([x, y], -1)

    cropped_xy = tf.image.stateless_random_crop(
        value=xy, size=(height, width, n_x_bands + n_y_bands), seed=seed
    )

    cropped_x = cropped_xy[:, :, :-n_y_bands]
    cropped_y = tf.cast(cropped_xy[:, :, -n_y_bands:], y_type)

    # explicit reshape to avoid
    # ValueError: as_list() is not defined on an unknown TensorShape.
    # which is thrown by model.fit
    new_x = tf.reshape(cropped_x, (height, width, n_x_bands))
    new_y = tf.squeeze(tf.reshape(cropped_y, (height, width, n_y_bands)))
    return new_x, new_y


def crop_data(x, y, height, width):
    """
    Creates a random seed then calls _crop_data
    """
    seed = RNG.make_seeds(2)[0]
    return _crop_data(x, y, height, width, seed)


def get_files_as_dataset(patterns, shuffle):
    if not isinstance(patterns, list):
        patterns = [patterns]

    if '*' in patterns[0]:  # patterns use unix style file expansion
        files = tf.data.Dataset.list_files(patterns[0], shuffle=shuffle)
        for p in patterns[1:]:
            files = files.concatenate(
                tf.data.Dataset.list_files(p, shuffle=shuffle)
            )
    else:  # patterns are complete file names
        files = tf.data.Dataset.list_files(patterns, shuffle=shuffle)

    return tf.data.TFRecordDataset(files, compression_type='GZIP')


def get_dataset(patterns, shape, image_bands, annotation_bands,
                batch_size=64, shuffle=False, repeat=False, augment=False,
                desired_shape=None):
    """
    patterns (str/str list):     Files to create dataset from. Can either be
                                 complete filepaths or unix style patterns.
    shape (int tuple):           Shape of each patch without band dimension.
                                 This is the shape of the patches as stored in
                                 the TFRecord, this can be shrunk by setting
                                 desired shape.
    image_bands (str list):      Name of each band to use in model input.
    annotation_bands (str list): Name of each band to use in ground truth.
    shuffle (bool):              if true shuffle dataset with buffer size 500.
    repeat (bool):               if true dataset repeates infinitely.
    prefetch (bool):             if true the dataset is prefetched with
                                 AUTOTUNE buffer.
    augment (bool):              If true the data is augmented with random
                                 flips etc...
    desired_shape (int tuple):   Desired shape of the final output, if given
                                 and different than shape each patch will be
                                 randomly cropped to match this shape. Should
                                 just be (height, width) i.e. no band dimension

    Returns a tf.data.TFRecordDataset
    """

    if not isinstance(image_bands, list):
        image_bands = [image_bands]
    if not isinstance(annotation_bands, list):
        if annotation_bands is not None:
            annotation_bands = [annotation_bands]

    # ensure band names are all valid
    for b in image_bands + annotation_bands:
        try:
            assert b in ALL_BANDS
        except AssertionError as E:
            raise ValueError(f'invalid band name: {b}') from E

    dataset = get_files_as_dataset(patterns, shuffle)

    dataset = dataset.map(
        lambda x: parse(x, shape, image_bands, annotation_bands),
        num_parallel_calls=AUTOTUNE
    )

    dataset = dataset.cache()

    if shuffle:
        dataset = dataset.shuffle(1000)

    if desired_shape is not None:
        if desired_shape[0] < shape[0] or desired_shape[1] < shape[1]:
            dataset = dataset.map(
                lambda x, y: crop_data(x, y, *desired_shape),
                num_parallel_calls=AUTOTUNE
            )

    if augment:
        dataset = dataset.map(augment_data, num_parallel_calls=AUTOTUNE)

    dataset = dataset.batch(batch_size)

    if repeat:
        dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset
