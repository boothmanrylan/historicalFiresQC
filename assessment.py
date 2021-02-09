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

def normalize_confusion_matrix(cm):
    cm = tf.cast(cm, tf.float32)
    class_counts = tf.math.reduce_sum(cm, 1, keepdims=True)
    class_counts = tf.where(class_counts == 0, 1, class_counts)
    return cm / class_counts

def plot_confusion_matrix(cm, xlabels, ylabels=None):
    # transpose confusion matrix to follow remote sensing convetions
    cm = tf.transpose(cm)

    if ylabels is None:
        ylabels = xlabels
    heatmap = sns.heatmap(cm, annot=True, xticklabels=xlabels,
                          yticklabels=ylabels)
    heatmap.set_ylabel('Predicted Class')
    heatmap.set_xlabel('True Class')
    heatmap.set_title('Confusion Matrix')
    plt.show()

def _errors(cm, classes, axis):
    n = len(classes)

    class_counts = tf.reduce_sum(cm, axis, keepdims=True)

    incorrect = cm * tf.math.abs(tf.eye(n) - 1)
    incorrect = tf.reduce_sum(incorrect, axis, keepdims=True)

    errors = tf.squeeze(incorrect / class_counts)
    return errors
    # errors = list(errors.numpy())
    # return dict(zip(classes, errors))

def errors_of_omission(cm, classes):
    return _errors(cm, classes, 0)

def errors_of_comission(cm, classes):
    return _errors(cm, classes, 1)

def confusion_matrix_with_errors(model, dataset, classes):
    cm = confusion_matrix(model, dataset, len(classes)) # needs to be transposed
    eoo = errors_of_omission(cm, classes) # goes on bottom
    eoc = errors_of_comission(cm, classes) # goes on right with NaN appended

    cm = tf.transpose(cm)

    cm_eoo = tf.stack([cm, eoo])
    eoc_nan = tf.stack([eoc, np.NaN])

    output = tf.stack([cm_eoo, eoc_nan], axis=1)

    return output



def acc(cm, classes):
    correct = cm * tf.eye(classes, classes)
    return tf.reduce_sum(correct) / tf.reduce_sum(cm)

def avg_class_acc(cm, classes):
    correct = cm * tf.eye(classes, classes)
    class_accs = tf.reduce_sum(correct, 1)
    return tf.reduce_mean(class_accs)

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
    for images, references in dataset:
        if model is not None:
            predictions = tf.argmax(model(images, training=False), -1)
        else:
            predictions = images
        predictions = tf.reshape(predictions, [-1])
        references = tf.reshape(references, [-1])

        # merge new and old burn classes
        predictions = tf.where(predictions > 4, 4, predictions)

        # drop all negative points in reference as they are not labelled
        mask = tf.reshape(tf.where(references >= 0), [-1])
        predictions = tf.gather(predictions, mask)
        references = tf.gather(references, mask)

        matrix += tf.math.confusion_matrix(references, predictions, num_classes)
    return matrix

def burn_age_reference_accuracy(model, dataset, inverse_burn_age_function,
                                max_burn_age):
    matrix = np.zeros((2, 2))
    for images, references in dataset:
        predictions = model(images, training=False)

        # convert burn age prediction into binary burn vs not burn class
        predictions = inverse_burn_age_function(predictions)
        predictions = tf.where(predictions < max_burn_age, 1, 0)

        # drop all negative points in references as they are not labelled
        mask = tf.reshape(tf.where(references >= 0), [-1])

        predictions = tf.gather(tf.reshape(predictions, [-1]), mask)
        references = tf.gather(tf.reshape(references, [-1]), mask)

        print(tf.math.reduce_min(references))
        print(tf.math.reduce_max(references))
        print(tf.math.reduce_mean(references))

        matrix += tf.math.confusion_matrix(references, predictions, 2)
    return matrix

def dated_burn_accuracy(model, dataset, num_classes, scale):
    """
    Dataset is expected to return tuples of image, references points
    All non-negative references are true burns
    The true burn reference point values are the age of the burn in days
    """
    try:
        assert scale in ['days', 'months', 'years']
    except AssertionError as E:
        raise ValueError(
            f'scale must be one of days, months, or years got {scale}'
        ) from E
    scale_factor = 1 if scale == 'days' else 30 if scale == 'months' else 365
    output = {}
    for images, references in dataset:
        references, burn_ages = references[:, :, :, 0], references[:, :, :, 1]
        if model is not None:
            predictions = tf.argmax(model(images, training=False), -1)
        else:
            predictions = images

        # flatten predictions and reference
        predictions = tf.reshape(predictions, [-1])
        references = tf.reshape(references, [-1])
        burn_ages = tf.reshape(burn_ages, [-1])

        # remove all non-burn points
        burnmask = tf.reshape(tf.where(references == 4), [-1])
        burn_ages = tf.gather(burn_ages, burnmask)
        predictions = tf.gather(predictions, burnmask)

        # merge new and old burn class
        predictions = tf.where(predictions > 4, 4, predictions)

        # get all of the unique burn ages
        ages, indices = tf.unique(burn_ages)

        # for every unique burn age determine how many of them were predicted
        # as each class
        for i, age in enumerate(ages.numpy()):
            # convert age to days, months, or years (floored)
            age = np.floor(age / scale_factor).astype(int)

            # get all the predictions for the current burn age
            age_i_mask = tf.reshape(tf.where(indices == i), [-1])
            age_i_preds = tf.gather(predictions, age_i_mask)

            # count the number of times burns of this age were labelled as each
            # class
            preds, _, counts = tf.unique_with_counts(age_i_preds)
            preds = tf.cast(preds, tf.int32)
            counts = tf.cast(counts, tf.int32)

            # its possible that some classes were never predicted, ensure they
            # are given a count of zero
            preds = tf.tensor_scatter_nd_add(
                tf.zeros(num_classes, tf.int32),
                tf.reshape(preds, (-1, 1)),
                counts
            )

            if age not in output.keys():
                output[age] = tf.zeros(num_classes, tf.int32)
            output[age] += preds

    # convert output to dict of lists
    for k, v in output.items():
        output[k] = list(v.numpy())
    return output

def plot_burn_accuracy_by_burn_age(model, dataset, class_labels,
                                   scale='days'):
    try:
        assert scale in ['days', 'months', 'years']
    except AssertionError as E:
        raise ValueError(
            f'scale must be one of days, months, or years. Got {scale}'
        ) from E
    num_classes = len(class_labels)
    results = dated_burn_accuracy(model, dataset, num_classes, scale)
    df = pd.DataFrame.from_dict(results)
    df.index = class_labels
    df /= df.sum()
    df = df.melt(ignore_index=False).reset_index()

    age_label = f'Burn Age ({scale.capitalize()})'

    df.columns = ['Predicted Class', age_label, '% Burns Predicted']

    sns.set_theme()
    palette = sns.crayon_palette(
        ['Forest Green', 'Navy Blue', 'Red']
    )
    hue_order = ['Land', 'Water', 'Burn']

    df = df[df['Predicted Class'].isin(hue_order)]

    sns.lmplot(x=age_label, y='% Burns Predicted',
               hue='Predicted Class', data=df, palette=palette,
               hue_order=hue_order, height=4, aspect=1.75)

def accuracy_assessment(matrix, labels):
    data = {
        label: list(matrix[i].numpy().astype(int))
        for i, label in enumerate(labels)
    }

    df = pd.DataFrame.from_dict(data)
    df.index = labels

    if 'None' in labels:
        df = df.drop('None')
        df = df.drop('None', axis=1)
        labels.remove('None')

    col_total = df.sum()
    row_total = df.sum(1)

    eye = pd.DataFrame(np.eye(len(labels)))
    eye.index = df.index
    eye.columns = df.columns

    eye *= df

    eoo = (col_total - eye.sum()) / col_total
    eoc = (row_total - eye.sum(1)) / row_total

    producers_accuracy = eye.sum() / col_total
    users_accuracy = eye.sum(1) / row_total

    df['Total'] = row_total
    df = df.append(col_total.rename('Total'))

    df['Errors of Commission'] = eoc
    df = df.append(eoo.rename('Errors of Omission'))

    df['User\'s Accuracy'] = users_accuracy
    df = df.append(producers_accuracy.rename('Producer\'s Accuracy'))

    df = df.round(decimals=4)

    return df

def burn_age_accuracy_assessment(model, dataset, inverse_burn_age_function, max_burn_age):
    matrix = burn_age_reference_accuracy(
        model, dataset, inverse_burn_age_function, max_burn_age
    )
    return accuracy_assessment(matrix, ['Not Burnt', 'Burnt'])

def classification_accuracy_assessment(model, dataset, labels):
    matrix = reference_accuracy(model, dataset, len(labels))
    return accuracy_assessment(matrix, labels)
