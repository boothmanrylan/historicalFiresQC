import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

def build_unet_model(input_shape, classes):
    """
    Constructs a basic UNET model.
    Based on tensorflow.org/tutorials/images/segmentation
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=None
    )

    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project'
    ]

    layers = [base_model.get_layer(name).output for name in layer_names]

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64,  3)
    ]

    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs

    skips = down_stack(x)
    x = skips[-1]

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        classes, 3, strides=2, padding='same',
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def weighted_loss(loss_fn, weights, **loss_args):
    '''
    loss_fn: any loss function, such as categorical_crossentropy
    weights: list of n elements where the ith element corresponds to the
             weight of the ith class.
    loss_args: any named arguments to pass to the loss_fn, such as from_logits
    '''
    def _weighted_loss(true, pred):
        loss = loss_fn(true, pred, **loss_args)
        return loss * tf.gather(weights, tf.cast(true, tf.int32))
    return _weighted_loss


def reference_point_loss(loss_fn, weights=None, alpha=1.0, beta=1.0, **args):
    """
    Include a term in the loss based on reference points.

    True values expected to have shape (x, y, 2) for input images with shape
    (x, y) i.e. as a channels last image where the channels are the default
    labels and the reference points repectively. Pixels that are not reference
    points should have negative values in the reference points channel.

    If weights are given, they are used to weight the term in the loss function
    based on the default labels.

    Applies loss_fn separately to bath (labels, pred) and (ref_points, pred).
    Returns the sum of both losses weighted by alpha and beta:
        (alpha * loss_fn(labels, pred)) + (beta * loss_fn(ref_points, pred))
    """
    def _ref_point_loss(true, pred):
        labels, ref_points = true[:, :, :, 0], true[:, :, :, 1]

        base_loss = loss_fn(labels, pred, **args)
        if weights is not None:
            base_loss *= tf.gather(weights, tf.cast(labels, tf.int32))

        # can't compute loss with values outside 0..n, but non reference points
        # have values < 0 therefore flip them all to 0 to calculate loss then
        # remove them later
        pos_ref_points = tf.where(
            ref_points < 0,
            tf.cast(0, ref_points.dtype),
            ref_points
        )

        ref_loss = loss_fn(pos_ref_points, pred, **args)

        # remove non reference points
        ref_loss *= tf.cast(tf.where(ref_points > 0, 1, 0), ref_loss.dtype)

        return (alpha * base_loss) + (beta * ref_loss)
    return _ref_point_loss

def basic_loss(loss_fn, **args):
    '''
    Here to make the basic loss more compatible with the other loss functions
    '''
    def _basic_loss(true, pred):
        return loss_fn(true, pred, **args)
    return _basic_loss
