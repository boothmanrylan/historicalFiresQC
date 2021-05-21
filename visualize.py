import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

colours = [
    (0, 0,    0), # black for no data
    (1, 1,    1), # white for clouds
    (0, 0,    1), # blue for water
    (0, 0.5,  0), # green for land (non-burnt)
    (1, 0,    0), # red for new burns
    (1, 0.65, 0)  # orange for old burns
]
bounds = [0, 1, 2, 3, 4, 5, 6]

normalized_colours = [
    (0, 0,   0), # black for no data
    (0, 0.5, 0), # green for land (non-burnt)
    (1, 0,   0), # red for burnS
]
normalized_bounds = [0, 1, 2, 3]

class_cmap = ListedColormap(colours)
class_norm = BoundaryNorm(bounds, class_cmap.N)
normalized_class_cmap = ListedColormap(normalized_colours)
normalized_class_norm = BoundaryNorm(normalized_bounds, normalized_class_cmap.N)

def false_colour_image(image):
    """
    Convert Landsat MSS image into a flase colour image where the red channel
    is Near Infrared 1 Band, the green channel is the Near Infrader 2 band and
    the blue channel is the red band. If stacked_image is True, then image
    contains two images of the same scene stacked on top of each other, return
    false colour images of both.
    """
    # input image must be a single MSS image with band order B4, B5, B6, B7,...
    assert image.ndim == 3

    fci = np.stack([
        image[:, :, 3],
        image[:, :, 2],
        image[:, :, 1]
    ], axis=-1)

    return fci

def greyscale_image(image, band):
    return image[:, :, band]

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

def visualize(dataset, model=None, num=20, greyscale_band=None):
    '''
    histogram should be None for no histogram or an int representing the numbe
    of classes in an anotation
    '''

    # outdated arguments
    stacked_image = False
    include_prev_burn_age = False
    include_prev_class = False
    max_annot = 2
    max_burn_age = 3650
    histogram=None
    normalized_data = True
    # end of outdated arguments

    if normalized_data:
        cmap = normalized_class_cmap
        norm = normalized_class_norm
    else:
        cmap = class_cmap
        norm = class_norm

    if greyscale_band is not None:
        image_fn = lambda im: greyscale_image(im, greyscale_band)
    else:
        image_fn = false_colour_image

    if max_annot is not None:
        cmap = 'gray'
        norm = None

    for images, annotations in dataset.take(num):
        image = tf.squeeze(images[0]).numpy()

        num_figs = 1 + annotations.shape[-1] if annotations.ndim == 4 else 2
        num_figs += stacked_image + include_prev_burn_age + include_prev_class

        if model is not None:
            num_figs += 1

        _, ax = plt.subplots(1, num_figs, figsize=(15, 30))

        fci = image_fn(image[:, :, :4])
        vmin, vmax = calculate_vmin_vmax(fci)
        ax[0].imshow(fci, vmin=vmin, vmax=vmax)
        ax[0].set_title('Input Patch')
        plot_offset = 1
        band_offset = 4
        if stacked_image:
            fci = image_fn(image[:, :, band_offset:band_offset + 4])
            vmin, vmax = calculate_vmin_vmax(fci)
            ax[plot_offset].imshow(fci, vmin=vmin, vmax=vmax)
            ax[plot_offset].set_title('Previous Patch')
            plot_offset += 1
            band_offset += 4
        if include_prev_burn_age:
            burn_age = np.squeeze(image[:, :, band_offset:band_offset + 1])
            ax[plot_offset].imshow(burn_age, vmin=0, vmax=max_burn_age)
            ax[plot_offset].set_title('Previous Burn Age')
            plot_offset += 1
            band_offset += 1
        if include_prev_class:
            prev_class = np.squeeze(image[:, :, band_offset:band_offset + 1])
            ax[plot_offset].imshow(prev_class, cmap=cmap, norm=norm,
                                   interpolation='nearest')
            ax[plot_offset].set_title('Previous Classification')
            plot_offset += 1
            band_offset += 1

        vmin, vmax = None, None
        if max_annot is not None:
            vmax = max_annot
            vmin = 0

        if annotations.ndim == 4:
            for i in range(annotations.shape[-1]):
                annotation = tf.squeeze(annotations[0, :, :, i]).numpy()
                hist_str = histogram_to_str(annotation, histogram)
                ax[i + plot_offset].imshow(annotation, vmin=vmin, vmax=vmax,
                                           cmap=cmap, interpolation='nearest',
                                           norm=norm)
                ax[i + plot_offset].set_title(f'Annotation {i + 1} {hist_str}')
        else:
            annotation = tf.squeeze(annotations[0]).numpy()
            hist_str = histogram_to_str(annotation, histogram)
            ax[plot_offset].imshow(annotation, vmin=vmin, vmax=vmax,
                                   cmap=cmap, interpolation='nearest',
                                   norm=norm)
            ax[plot_offset].set_title(f'Annotation {hist_str}')

        if model is not None:
            prediction = tf.argmax(
                model(images, training=False)[0], -1
            ).numpy()
            hist_str = histogram_to_str(prediction, histogram)
            ax[num_figs - 1].imshow(prediction, vmin=vmin, vmax=vmax,
                                    cmap=cmap, interpolation='nearest',
                                    norm=norm)
            ax[num_figs - 1].set_title(f'Model Prediction {hist_str}')
        plt.show()
