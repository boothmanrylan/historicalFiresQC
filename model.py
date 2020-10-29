import tensorflow as tf
import tensorflow_addons as tfa
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
        concat = tf.keras.layer.Concatenate()
        x = concat([x, skip])

    last = tf.keras.layers.Conv2DTranspose(
        classes, 3, strides=2, padding='same',
    )

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
