import tensorflow as tf
import numpy as np
from tensorflow_examples.models.pix2pix import pix2pix
from tensorflow.keras import backend as K

def build_unet_model(input_shape, classes):
    """
    Constructs a basic UNET model.
    Based on tensorflow.org/tutorials/images/segmentation
    """

    print('creating base model')
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights=None
    )
    print('done creating base model')

    layer_names = [
        'block_1_expand_relu',
        'block_3_expand_relu',
        'block_6_expand_relu',
        'block_13_expand_relu',
        'block_16_project'
    ]

    print('creating down stack')
    layers = [base_model.get_layer(name).output for name in layer_names]

    down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
    print('done creating down stack')

    up_stack = [
        pix2pix.upsample(512, 3),
        pix2pix.upsample(256, 3),
        pix2pix.upsample(128, 3),
        pix2pix.upsample(64,  3)
    ]

    print('creating input layer')
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = inputs
    print('done creating input layer')

    print('creating skips')
    skips = down_stack(x)
    x = skips[-1]
    print('done creating skips')

    print('reversing skips')
    skips = reversed(skips[:-1])
    print('done reversing skips')

    print('connecting layers')
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])
    print('done connecting layers')

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

    applies loss_fn separately to bath (labels, pred) and (ref_points, pred).
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

def no_burn_edge_loss(loss_fn, **args):
    '''
    dont use edges of burned regions when computing the loss.
    expects true values to have shape (x, y, 2) for input images with shape
    (x, y) true[:, :, 0] should be the true classification/regression values
    and true[:, :, 1] should be a matrix of 0s and 1s that is 0 on all the burn
    edges. args get pased to loss_fn.
    returns (loss_fn(true[:, :, 0], pred) * true[:, :, 1])
    '''
    def _no_burn_edge_loss(true, pred):
        assert true.shape[-1] == 2
        labels, burn_edge_mask = true[:, :, :, 0], true[:, :, :, 1]
        base_loss = loss_fn(labels, pred, **args)
        return base_loss * tf.cast(burn_edge_mask, base_loss.dtype)
    return _no_burn_edge_loss

class MeanIoUFromLogits(tf.keras.metrics.Metric):
    def __init__(self, num_classes, name=None, dtype=None):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes

        self.total_cm = self.add_weight(
            'total_confusion_matrix',
            shape=(num_classes, num_classes),
            initializer=tf.zeros_initializer,
            dtype=tf.float64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred, self._dtype)

        # convert from logits
        y_pred = tf.math.argmax(y_pred, axis=-1)

        if y_pred.shape.ndims > 1:
            y_pred = tf.reshape(y_pred, [-1])

        if y_true.shape.ndims > 1:
            y_true = tf.reshape(y_true, [-1])

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self._dtype)
            if sample_weight.shape.ndims > 1:
                sample_weight = tf.reshape(sample_weight, [-1])

        current_cm = tf.math.confusion_matrix(
            y_true, y_pred, self.num_classes, weights=sample_weight,
            dtype=tf.float64)
        return self.total_cm.assign_add(current_cm)

    def result(self):
        sum_over_row = tf.cast(
            tf.reduce_sum(self.total_cm, axis=0), self._dtype)
        sum_over_col = tf.cast(
            tf.reduce_sum(self.total_cm, axis=1), self._dtype)
        true_positives = tf.cast(
            tf.linalg.tensor_diag_part(self.total_cm), self._dtype)

        denominator = sum_over_row + sum_over_col - true_positives

        num_valid_entries = tf.reduce_sum(
            tf.cast(tf.math.not_equal(denominator, 0), self._dtype))

        iou = tf.math.divide_no_nan(true_positives, denominator)

        return tf.math.divide_no_nan(
            tf.reduce_sum(iou, name='mean_iou'), num_valid_entries)

    def reset_states(self):
        return K.set_value(
            self.total_cm, np.zeros((self.num_classes, self.num_classes)))

    def get_config(self):
        config = {'num_classes': self.num_classes}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
