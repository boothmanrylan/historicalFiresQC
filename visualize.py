import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

MAX_ANNOT = 2

COLOURS = [
    (0.5, 0.5, 0.5),  # grey for no data
    (0.5, 0.25, 0),   # brown for land (non-burnt)
    (1, 0, 0),        # red for burns
    (0, 1, 0),        # blue for water
    (0, 0, 0),        # black for shadow
    (1, 1, 1),        # white for clouds
]
BOUNDS = [0, 1, 2, 3, 4, 5, 6]

CLASS_CMAP = ListedColormap(COLOURS)
CLASS_NORM = BoundaryNorm(BOUNDS, CLASS_CMAP.N)


def false_colour_image(image):
    """
    Converts Landsat MSS image into a flase colour image where the red channel
    is Near Infrared 1 Band, the green channel is the Near Infrared 2 band and
    the blue channel is the red band.

    image is expected to have band order [B4, B5, B6, B7, ... ]
    """
    assert image.ndim == 3

    fci = np.stack([
        image[:, :, 3],
        image[:, :, 2],
        image[:, :, 1]
    ], axis=-1)

    return fci


def calculate_vmin_vmax(image, alpha=0.9):
    mean = np.mean(image)
    std = np.std(image)
    vmin = mean - (alpha / 2 * std)
    vmax = mean + (alpha / 2 * std)
    return vmin, vmax


def histogram_to_str(annotation, classes):
    if classes is not None:
        hist, bin_edges = np.histogram(annotation, range(classes + 1))
        output = ', '.join([f'{x}: {y}' for x, y in zip(bin_edges[:-1], hist)])
    else:
        output = None
    return output


def visualize(dataset, model=None, num=20):
    """
    Displays num (image, annotation) pairs from dataset.
    If model is given also display the result of the model on each image.
    """
    for images, annotations in dataset.take(num):
        image = tf.squeeze(images[0]).numpy()

        num_figs = 2

        if model is not None:
            num_figs += 1

        _, ax = plt.subplots(1, num_figs, figsize=(15, 30))

        fci = false_colour_image(image[:, :, :4])
        vmin, vmax = calculate_vmin_vmax(fci)
        ax[0].imshow(fci, vmin=vmin, vmax=vmax)
        ax[0].set_title('Input Patch')
        plot_offset = 1

        annotation = tf.squeeze(annotations[0]).numpy()
        hist_str = histogram_to_str(annotation, MAX_ANNOT + 1)
        ax[plot_offset].imshow(annotation, vmin=0, vmax=MAX_ANNOT,
                               cmap=CLASS_CMAP, interpolation='nearest',
                               norm=CLASS_NORM)
        ax[plot_offset].set_title(f'Annotation {hist_str}')

        if model is not None:
            probs = model(images, training=False)
            prediction = tf.argmax(probs[0], -1).numpy()
            hist_str = histogram_to_str(prediction, MAX_ANNOT + 1)
            ax[num_figs - 1].imshow(prediction, vmin=0, vmax=MAX_ANNOT,
                                    cmap=CLASS_CMAP, interpolation='nearest',
                                    norm=CLASS_NORM)
            ax[num_figs - 1].set_title(f'Model Prediction {hist_str}')
        plt.show()
