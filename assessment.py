import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def confusion_matrix(model, dataset, classes=5):
    matrix = tf.zeros((classes, classes), dtype=tf.int32)
    for images, labels in dataset:
        predictions = tf.argmax(model(images, training=False), -1)

        # flatten predictions and labels
        predictions = tf.reshape(predictions, [-1])
        labels = tf.reshape(labels, [-1])

        matrix += tf.math.confusion_matrix(labels, predictions, classes)
    return matrix

def normalize_confusion_matrix(confusion_matrix):
    confusion_matrix = tf.cast(confusion_matrix, tf.float32)
    class_counts = tf.math.reduce_sum(confusion_matrix, 1, keepdims=True)
    class_counts = tf.where(class_counts == 0, 1, class_counts)
    return confusion_matrix / class_counts

def plot_confusion_matrix(confusion_matrix, xlabels, ylabels=None):
    if ylabels is None:
        ylabels = xlabels
    heatmap = sns.heatmap(confusion_matrix, annot=True, xticklabels=xlabels,
                          yticklabels=ylabels)
    heatmap.set_ylabel('Predicted Class')
    heatmap.set_xlabel('"True" Class')
    heatmap.set_title('Confusion Matrix')
    plt.show()

def _errors(confusion_matrix, classes, axis):
    n = len(classes)

    class_counts = tf.reduce_sum(confusion_matrix, axis, keepdims=True)

    incorrect = confusion_matrix * tf.math.abs(tf.eye(n) - 1)
    incorrect = tf.reduce_sum(incorrect, axis, keepdims=True)

    errors = tf.squeeze(incorrect / class_counts)
    errors = list(errors.numpy())

    return dict(zip(classes, errors))

def errors_of_omission(confusion_matrix, classes):
    return _errors(confusion_matrix, classes, 0)

def error_of_comission(confusion_matrix, classes):
    return _errors(confusion_matrix, classes, 1)

def acc(confusion_matrix, classes):
    correct = confusion_matrix * tf.eye(classes, classes)
    return tf.reduce_sum(correct) / tf.reduce_sum(confusion_matrix)

def avg_class_acc(confusion_matrix, classes):
    correct = confusion_matrix * tf.eye(classes, classes)
    class_accs = tf.reduce_sum(correct, 1)
    return tf.reduce_avg(avg_acc)

def split_class_cm(model, dataset, true_classes=6, predicted_classes=5):
    matrix = np.zeros((true_classes, predicted_classes))
    for images, labels in dataset:
        predictions = tf.argmax(model(images, training=False), -1)

        predictions = tf.reshape(predictions, [-1])
        labels = tf.reshape(labels, [-1])

        for t in range(true_classes):
            for p in range(predicted_classes):
                curr = tf.logical_and(labels == t, predictions == p)
                matrix[t, p] += tf.reduce_sum(tf.cast(curr, tf.int32))
    return matrix

def reference_accuracy(model, dataset, num_classes):
    matrix = np.zeros((num_classes, num_classes))
    for image, reference in dataset:
        predictions = tf.argmax(model(image, training=False), -1)
        predictions = tf.reshape(predictions, [-1])
        references = tf.reshape(reference, [-1])

        # drop all 0 points in reference as they are not labelled
        mask = tf.reshape(tf.where(references != 0), [-1])
        predictions = tf.gather(predictions, mask)
        references = tf.gather(references, mask)

        matrix += tf.math.confusion_matrix(references, predictions, num_classes)
    return matrix


def dated_burn_accuracy(model, dataset, num_classes):
    """
    Dataset is expected to return tuples of image, references points
    All non-negative references are true burns
    The true burn reference point values are the age of the burn in days
    """
    output = {}
    for images, references in dataset:
        predictions = tf.argmax(model(images, training=False), -1)

        # flatten predictions and reference
        predictions = tf.reshape(predictions, [-1])
        references = tf.reshape(references, [-1])

        # remove all non-burn points
        burnmask = tf.reshape(tf.where(references >= 0), [-1])
        predictions = tf.gather(predictions, burnmask)
        references = tf.gather(references, burnmask)

        # get all of the unique burn ages
        ages, indices = tf.unique(references)

        # for every unique burn age determine how many of them were predicted
        # as each class
        for i, age in enumerate(ages.numpy()):
            # get all the predictions for the current burn age
            age_i_mask = tf.reshape(tf.where(indices == i), [-1]))
            age_i_preds = tf.gather(predictions, age_i_mask)

            # count the number of times burns of this age were labelled as each
            # class
            preds, _, counts = tf.unique_with_counts(age_i_preds)

            # its possible that some classes were never predicted, ensure they
            # are given a count of zero
            preds = tf.tensor_scatter_nd_add(
                tf.zeros(num_classes),
                tf.reshape(preds, (-1, 1)),
                counts
            )

            if age not in output.keys():
                output[age] = tf.zeros(num_classes)
            output[age] += preds

    # convert output to dict of lists
    for k in output.keys():
        output[k] = list(output[k].numpy)
    return output

def plot_burn_accuracy_by_burn_age(model, dataset, num_classes):
    if num_classes == 5:
        class_names = ['None', 'Cloud', 'Water', 'Land', 'Burn']
    elif num_classes == 6:
        class_names = ['None', 'Cloud', 'Water', 'Land', 'New Burn', 'Old Burn']
    else:
        raise ValueError('bad num_classses')
    results = dated_burn_accuracy(model, dataset, num_classes)
    df = pd.DataFrame.from_dict(results)
    df.index = class_names
    df = df.divide(df.sum(axis=1), axis=1)
    longdf = df.melt(ignore_index=False).reset_index()
    longdf.columns = ['Predicted Class', 'Burn Age', 'Percent']
    graph = sns.countplot(x='Burn Age', y='Percent', hue='Predicted Class',
            data=longdf)
    plt.show()


