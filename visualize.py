import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

colours = [
    (0, 0,    0), # white for clouds
    (1, 1,    1), # black for no data
    (0, 0,    1), # blue for water
    (0, 0.5,  0), # green for land (non-burnt)
    (1, 0,    0), # red for new burns
    (1, 0.65, 0)  # orange for old burns
]

cmap = ListedColormap(colours)
bounds = [0, 1, 2, 3, 4, 5, 6]
norm = BoundaryNorm(bounds, cmap.N)

def show_predictions(dataset, model, num):
    for images, true_annotations in dataset.take(num):
        image = tf.squeeze(images[0]).numpy()
        true_annotation = tf.squeeze(true_annotations[0]).numpy()
        prediction = tf.argmax(model(images, training=False)[0], -1).numpy()

        # select and order bands for false colour image
        false_colour_image = np.stack([
            image[:, :, -1], # R = Near Infrared 1
            image[:, :, -2], # G = Near Infrared 2
            image[:, :, -3]  # B = Red
        ], axis=-1)

        # calculate optimal range of values for false colour image
        fci_mean = np.mean(false_colour_image)
        fci_std = np.std(false_colour_image)
        vmin = fci_mean - (0.45 * fci_std)
        vmax = fci_mean + (0.45 * fci_std)

        f, ax = plt.subplots(1, 4, figsize=(15, 30))
        ax[0].imshow(false_colour_image, vmin=vmin, vmax=vmax)
        ax[0].set_title('Input Patch')

        ax[1].imshow(true_annotation, vmin=0, vmax=len(colours), cmap=cmap,
                     interpolation='nearest', origin='lower', norm=norm)
        ax[1].set_title('Ground "Truth"')

        ax[2].imshow(prediction, vmin=0, vmax=len(colours), cmap=cmap,
                     interpolation='nearest', origin='lower', norm=norm)
        ax[2].set_title('Model Prediction')

        ax[3].imshow(prediction == true_annotation)
        ax[3].set_title('Misclassifications')
