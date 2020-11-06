import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

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
    return confusion_matrix / class_counts

def plot_confusion_matrix(confusion_matrix, xlabels, ylabels=None):
    if ylabels is None:
        ylabels = xlabels
    heatmap = sns.heatmap(confusion_matrix, annot=True, xticklabels=xlabels,
                          yticklabels=ylabels)
    heatmap.set_xlabel('Predicted Class')
    heatmap.set_ylabel('"True" Class')
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
