import gc
import json
import os
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
import numpy as np
import data as Data
import model as Model


class ClearMemory(Callback):
    """
    From: github.com/tensorflow/tensorflow/issues/31312#issuecomment-821809246
    """
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()


def train(model, model_path, steps_per_epoch, epochs, store_model,
          learning_rate, train_dataset):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path, save_weights_only=True,
        save_freq=steps_per_epoch * int(epochs / 5)
    )
    callbacks = None
    if store_model:
        callbacks = [checkpoint]

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  run_eagerly=True)
    model.fit(
        train_dataset, epochs=epochs, verbose=1,
        steps_per_epoch=steps_per_epoch, callbacks=callbacks
    )


def predict(bucket, model, test_folder, predictions_folder, shape, image_bands,
            overlap):
    uploads = []
    mixer_files = tf.io.gfile.glob(
        os.path.join(bucket, test_folder, '*.json')
    )

    for m in mixer_files:
        with tf.io.gfile.GFile(m, 'r') as f:
            mixer = json.loads(f.read())
        patches = mixer['totalPatches']

        pattern = m.replace('.json', '*.tfrecord.gz')
        pattern = pattern.replace('mixer', '')
        tfrecords = tf.io.gfile.glob(pattern)
        tfrecords.sort()

        dataset = Data.get_dataset(
            patterns=tfrecords, shape=shape, image_bands=image_bands,
            annotation_bands=None, batch_size=1, shuffle=False
        )

        predictions = model.predict(dataset, steps=patches, verbose=1)

        filename = m.replace('.json', '-results.tfrecord')
        filename = filename.replace(test_folder, predictions_folder)
        if filename[0] == '/':  # remove erroneous /
            filename = filename[1:]

        uploads.append({
            'mixer': m,
            'image': filename
        })

        with tf.io.TFRecordWriter(filename) as writer:
            for pred in predictions:
                k = int(overlap / 2)
                pred = pred[k:-k, k:-k]
                value = np.argmax(pred, -1).flatten()
                feature = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=value)
                )

                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={'class': feature}
                    )
                )

                writer.write(example.SerializeToString())
    return uploads


def main(bucket='boothmanrylan', data_pattern='rylansPicks*.tfrecord.gz',
         model_path='rylansPicksModel', store_model=True,
         shape=(160, 160), desired_shape=None, batch_size=32,
         learning_rate=1e-4, epochs=100, steps_per_epoch=100,
         train_model=False, load_model=False, make_predictions=False,
         shuffle=False, repeat=False, augment=False, adjust_brightness=False,
         test_folder='historicalFiresQCMaskedData',
         predictions_folder='rylansPicks', overlap=32, image_bands=None):

    if image_bands is None:
        image_bands = Data.ALL_BANDS
        try:
            image_bands.remove('class')
        except ValueError:
            pass

    annotation_bands = ['class']

    model_path = os.path.join(bucket, model_path)

    channels = len(image_bands)
    classes = 3  # none, not-burn, burn

    if desired_shape is not None:
        try:
            assert desired_shape[0] <= shape[0]
            assert desired_shape[1] <= shape[1]
        except AssertionError as E:
            raise ValueError("if given, desired_shape must be <= shape") from E
        model_shape = desired_shape
    else:
        model_shape = shape

    model = Model.build_unet_model(
        input_shape=(*model_shape, channels), classes=classes
    )

    if load_model:
        model.load_weights(model_path)

    train_dataset = Data.get_dataset(
        patterns=os.path.join(bucket, data_pattern), shape=shape,
        image_bands=image_bands, annotation_bands=annotation_bands,
        batch_size=batch_size, shuffle=shuffle, repeat=repeat,
        augment=augment, adjust_brightness=adjust_brightness,
        desired_shape=desired_shape
    )

    if train_model:
        train(model, model_path, steps_per_epoch, epochs, store_model,
              learning_rate, train_dataset)

    uploads = None
    if make_predictions:
        uploads = predict(bucket, model, test_folder, predictions_folder,
                          shape, image_bands, overlap)

    return model, train_dataset, uploads
