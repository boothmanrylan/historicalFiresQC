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

def false_colour_image(image):
    """
    Convert Landsat MSS image into a flase colour image where the red channel
    is Near Infrared 1 Band, the green channel is the Near Infrader 2 band and
    the blue channel is the red band.
    """
    return np.stack([
        image[:, :, -1],
        image[:, :, -2],
        image[:, :, -3]
    ], axis=-1)

def calculate_vmin_vmax(image, alpha=0.9):
    mean = np.mean(image)
    std = np.std(image)
    vmin = mean - (alpha / 2 * std)
    vmax = mean + (alpha / 2 * std)
    return vmin, vmax

def visualize(dataset, model=None, num=20):
    for data in dataset.take(num):
        num_figs = len(data)
        if model is not None:
            num_figs += 1
        f, ax = plt.subplots(1, num_figs, figsize=(15, 30))

        image = tf.squeeze(data[0][0]).numpy()
        fci = false_colour_image(image)
        vmin, vmax = calculate_vmin_vmax(fci)
        ax[0].imshow(fci, vmin=vmin, vmax=vmax)
        ax[0].set_title('Input Patch')

        for i, annotations in enumerate(data[1:]):
            annotation = tf.squeeze(annotations[0]).numpy()
            ax[i + 1].imshow(annotation, vmin=0, vmax=len(colours), cmap=cmap,
                             interpolation='nearest', norm=norm)
            ax[i + 1].set_title(f'Annotation {i + 1}')

        if model is not None:
            prediction = tf.argmax(
                model(data[0], training=False)[0], -1
            ).numpy()
            ax[num_figs - 1].imshow(prediction, vmin=0, vmax=len(colours), cmap=cmap,
                                    interpolation='nearest', norm=norm)
            ax[num_figs - 1].set_title('Model Prediction')
