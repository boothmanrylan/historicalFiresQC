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

split_burnt_cmap = ListedColormap(colours)
combined_burnt_cmap = ListedColormap(colours[:-1])
split_burnt_norm = BoundaryNorm(bounds, split_burnt_cmap.N)
combined_burnt_norm = BoundaryNorm(bounds[:-1], combined_burnt_cmap.N)

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

def show_predictions(dataset, model, num, combined_burnt=False):
    num_classes = len(colours)
    cmap = split_burnt_cmap
    norm = split_burnt_norm
    if combined_burnt:
        num_classes -= 1
        cmap = combined_burnt_cmap
        norm = combined_burnt_norm
    for images, true_annotations in dataset.take(num):
        image = tf.squeeze(images[0]).numpy()
        true_annotation = tf.squeeze(true_annotations[0]).numpy()
        prediction = tf.argmax(model(images, training=False)[0], -1).numpy()

        fci = false_colour_image(image)
        vmin, vmax = calculate_vmin_vmax(fci)

        f, ax = plt.subplots(1, 4, figsize=(15, 30))
        ax[0].imshow(fci, vmin=vmin, vmax=vmax)
        ax[0].set_title('Input Patch')

        ax[1].imshow(true_annotation, vmin=0, vmax=num_classes, cmap=cmap,
                     interpolation='nearest', norm=norm)
        ax[1].set_title('Ground "Truth"')

        ax[2].imshow(prediction, vmin=0, vmax=num_classes, cmap=cmap,
                     interpolation='nearest', norm=norm)
        ax[2].set_title('Model Prediction')

        ax[3].imshow(prediction == true_annotation)
        ax[3].set_title('Misclassifications')

def show_dataset(dataset, num, combined_burnt=False):
    num_classes = len(colours)
    cmap = split_burnt_cmap
    norm = split_burnt_norm
    if combined_burnt:
        num_classes -= 1
        cmap = combined_burnt_cmap
        norm = combined_burnt_norm
    for images, annotations in dataset.take(num):
        image = tf.squeeze(images[0]).numpy()
        annotation = tf.squeeze(annotations[0]).numpy()

        fci = false_colour_image(image)
        vmin, vmax = calculate_vmin_vmax(fci)

        f, ax = plt.subplots(1, 4, figsize=(15, 30))
        ax[0].imshow(fci, vmin=vmin, vmax=vmax)
        ax[0].set_title('Input Image')

        ax[1].imshow(annotation, vmin=0, vmax=num_classes, cmap=cmap,
                     interpolation='nearest', norm=norm)
        ax[1].set_titel('Annotation')
