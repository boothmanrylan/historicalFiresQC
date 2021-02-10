import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

colours = [
    (0, 0,    0), # white for clouds
    (1, 1,    1), # black for no data
    (0, 0,    1), # blue for water
    (0, 0.5,  0), # green for land (non-burnt)
    (1, 0,    0), # red for new burns
    (1, 0.65, 0)  # orange for old burns
]
bounds = [0, 1, 2, 3, 4, 5, 6]

cmap = ListedColormap(colours)
norm = BoundaryNorm(bounds, cmap.N)

def false_colour_image(image, stacked_image=False):
    """
    Convert Landsat MSS image into a flase colour image where the red channel
    is Near Infrared 1 Band, the green channel is the Near Infrader 2 band and
    the blue channel is the red band. If stacked_image is True, then image
    contains two images of the same scene stacked on top of each other, return
    false colour images of both.
    """
    image1 = np.stack([
        image[:, :, 3],
        image[:, :, 2],
        image[:, :, 1]
    ], axis=-1)

    if stacked_image:
        image2 = np.stack([
            image[:, :, 7],
            image[:, :, 6],
            image[:, :, 5]
        ], axis=-1)
        output = image1, image2
    else:
        output = image1
    return output

def calculate_vmin_vmax(image, alpha=0.9):
    mean = np.mean(image)
    std = np.std(image)
    vmin = mean - (alpha / 2 * std)
    vmax = mean + (alpha / 2 * std)
    return vmin, vmax

def visualize(dataset, model=None, num=20, stacked_image=False, max_annot=None):
    for images, annotations in dataset.take(num):
        num_figs = 1 + annotations.shape[-1] if annotations.ndim == 4 else 2
        if model is not None:
            num_figs += 1
        if stacked_image:
            num_figs += 1
        _, ax = plt.subplots(1, num_figs, figsize=(15, 30))

        image = tf.squeeze(images[0]).numpy()
        fcis = false_colour_image(image, stacked_image)
        if not stacked_image:
            vmin, vmax = calculate_vmin_vmax(fcis)
            ax[0].imshow(fcis, vmin=vmin, vmax=vmax)
            ax[0].set_title('Input Patch')
        else:
            vmin, vmax = calculate_vmin_vmax(fcis[0])
            ax[0].imshow(fcis[0], vmin=vmin, vmax=vmax)
            ax[0].set_title('Current Patch')
            vmin, vmax = calculate_vmin_vmax(fcis[1])
            ax[1].imshow(fcis[1], vmin=vmin, vmax=vmax)
            ax[1].set_title('Previous Patch')

        vmax = len(colours) if max_annot is None else max_annot

        offset = 2 if stacked_image else 1
        if annotations.ndim == 4:
            for i in range(annotations.shape[-1]):
                annotation = tf.squeeze(annotations[0, :, :, i]).numpy()
                ax[i + offset].imshow(annotation, vmin=0, vmax=vmax,
                                      cmap=cmap, interpolation='nearest',
                                      norm=norm)
                ax[i + offset].set_title(f'Annotation {i + 1}')
        else:
            annotation = tf.squeeze(annotations[0]).numpy()
            ax[offset].imshow(annotation, vmin=0, vmax=vmax,
                              cmap=cmap, interpolation='nearest',
                              norm=norm)
            ax[offset].set_title('Annotation')

        if model is not None:
            prediction = tf.argmax(
                model(images, training=False)[0], -1
            ).numpy()
            ax[num_figs - 1].imshow(prediction, vmin=0, vmax=vmax,
                                    cmap=cmap, interpolation='nearest',
                                    norm=norm)
            ax[num_figs - 1].set_title('Model Prediction')
