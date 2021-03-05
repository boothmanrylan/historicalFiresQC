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

class_cmap = ListedColormap(colours)
class_norm = BoundaryNorm(bounds, class_cmap.N)

def false_colour_image(image):
    """
    Convert Landsat MSS image into a flase colour image where the red channel
    is Near Infrared 1 Band, the green channel is the Near Infrader 2 band and
    the blue channel is the red band. If stacked_image is True, then image
    contains two images of the same scene stacked on top of each other, return
    false colour images of both.
    """
    # input image must be a single MSS image
    assert image.ndim == 3 and image.shape[-1] == 4

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

def visualize(dataset, model=None, num=20, stacked_image=False,
              include_prev_burn_age=False, include_prev_class=False,
              max_annot=None, max_burn_age=3650):
    for images, annotations in dataset.take(num):
        image = tf.squeeze(images[0]).numpy()

        num_figs = 1 + annotations.shape[-1] if annotations.ndim == 4 else 2
        num_figs += stacked_image + include_prev_burn_age + include_prev_class

        if model is not None:
            num_figs += 1

        _, ax = plt.subplots(1, num_figs, figsize=(15, 30))

        fci = false_colour_image(image[:, :, :4])
        vmin, vmax = calculate_vmin_vmax(fci)
        ax[0].imshow(fci, vmin=vmin, vmax=vmax)
        ax[0].set_title('Input Patch')
        plot_offset = 1
        band_offset = 4
        if stacked_image:
            fci = false_colour_image(image[:, :, band_offset:band_offset + 4])
            vmin, vmax = calculate_vmin_vmax(fci)
            ax[plot_offset].imshow(fci, vmin=vmin, vmax=vmax)
            ax[plot_offset].set_title('Previous Patch')
            plot_offset += 1
            band_offset += 4
        if include_prev_burn_age:
            burn_age = image[:, :, band_offset:band_offset + 1]
            ax[plot_offset].imshow(burn_age, vmin=0, vmax=max_burn_age)
            ax[plot_offset].set_title('Previous Burn Age')
            plot_offset += 1
            band_offset += 1
        if include_prev_class:
            prev_class = image[:, :, band_offset:band_offset + 1]
            ax[plot_offset].imshow(prev_class, cmap=class_cmap, norm=norm_cmap,
                                   interpolation='nearest')
            ax[plot_offset].set_title('Previous Classification')
            plot_offset += 1
            band_offset += 1

        vmax, vmin = None, None
        cmap = class_cmap
        norm = class_norm
        if max_annot is not None:
            vmax = max_annot
            vmin = 0
            cmap = 'gray'
            norm = None

        if annotations.ndim == 4:
            for i in range(annotations.shape[-1]):
                annotation = tf.squeeze(annotations[0, :, :, i]).numpy()
                ax[i + plot_offset].imshow(annotation, vmin=vmin, vmax=vmax,
                                           cmap=cmap, interpolation='nearest',
                                           norm=norm)
                ax[i + plot_offset].set_title(f'Annotation {i + 1}')
        else:
            annotation = tf.squeeze(annotations[0]).numpy()
            ax[plot_offset].imshow(annotation, vmin=vmin, vmax=vmax,
                                   cmap=cmap, interpolation='nearest',
                                   norm=norm)
            ax[plot_offset].set_title('Annotation')

        if model is not None:
            prediction = tf.argmax(
                model(images, training=False)[0], -1
            ).numpy()
            ax[num_figs - 1].imshow(prediction, vmin=vmin, vmax=vmax,
                                    cmap=cmap, interpolation='nearest',
                                    norm=norm)
            ax[num_figs - 1].set_title('Model Prediction')
        plt.show()
