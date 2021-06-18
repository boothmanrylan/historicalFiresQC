import json
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import data as Data
import model as Model

def main(bucket='boothmanrylan', data_pattern='rylansPicks*.tfrecord.gz',
         modelA='rylansPicksModel', modelB=None, store_model=True,
         shape=(160, 160), batch_size=32,
         learning_rate=1e-4, epochs=100, steps_per_epoch=100,
         train_model=False, load_model=False,
         min_burn_percent=None, percent_burn_free=None, predict=False,
         test_folder='historicalFiresQCMaskedData',
         predictions_folder='rylansPicks', dataset_options=None, overlap=32,
         image_bands=None, nested=False):
    print('Starting main')

    if image_bands is None:
        image_bands = ['B4', 'B4', 'B6', 'B7', 'TCA', 'bai']

    annotation_bands = ['class']

    train_filter = lambda x, y: Data.filter_mostly_burnt(x, y, 2, min_burn_percent)
    if min_burn_percent == 0:
        train_filter = False

    if modelB is None:
        modelB = modelA

    load_model_path = os.path.join(bucket, modelA)
    store_model_path = os.path.join(bucket, modelB)

    channels = len(image_bands)
    classes = 3 # none, not-burn, burn

    print('building model')
    model = Model.build_unet_model(
        input_shape=(*shape, channels), classes=classes
    )
    print('done builing model')

    if load_model:
        print(f'loading model from {load_model_path}')
        model.load_weights(load_model_path)
        print('done loading model')
    else:
        print('not loading model')

    print('building train dataset')
    if dataset_options is None:
        dataset_options = {}

    train_dataset = Data.get_dataset(
        **dataset_options,
        patterns=os.path.join(bucket, data_pattern),
        shape=shape,
        image_bands=image_bands,
        annotation_bands=annotation_bands,
        batch_size=batch_size,
        filters=train_filter,
        repeat=True,
        percent_burn_free=percent_burn_free,
        burn_class=2,
        train=True
    )
    print('done building train dataset')

    if train_model:
        print('training model')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=store_model_path, save_weights_only=True,
            save_freq=steps_per_epoch * int(epochs / 5)
        )
        callbacks = None
        if store_model:
            print(f'storing checkpoints at {store_model_path}')
            callbacks = [checkpoint]
        else:
            print('not saving model weights')

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(
            train_dataset, epochs=epochs, verbose=1,
            steps_per_epoch=steps_per_epoch, callbacks=callbacks
        )
    else:
        print('not training model')

    uploads = []
    if predict and not nested:
        print(f'storing predictions for {test_folder} to {predictions_folder}')
        mixer_files = tf.io.gfile.glob(os.path.join(bucket, test_folder, '*.json'))

        for m in mixer_files:
            with tf.io.gfile.GFile(m, 'r') as f:
                mixer = json.loads(f.read())
            patches = mixer['totalPatches']

            pattern = m.replace('.json', '*.tfrecord.gz')
            pattern = pattern.replace('mixer', '')
            tfrecords = tf.io.gfile.glob(pattern)
            tfrecords.sort()

            print(f'building dataset for {pattern}')
            dataset = Data.get_dataset(
                patterns=tfrecords, shape=shape, image_bands=image_bands,
                annotation_bands=None, batch_size=1, filters=False,
                shuffle=False, train=False, burn_class=2
            )
            print('done building dataset')

            predictions = model.predict(dataset, steps=patches, verbose=1)

            filename = m.replace('.json', '-results.tfrecord')
            filename = filename.replace(test_folder, predictions_folder)
            if filename[0] == '/': # remove erroneous /
                filename = filename[1:]

            print(f'Writing results for {m} to {filename}')

            with tf.io.TFRecordWriter(filename) as writer:
                patch = 1
                for pred in predictions:
                    k = int(overlap / 2)
                    pred = pred[k:-k, k:-k]
                    value = np.argmax(pred, -1).flatten()
                    feature = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=value)
                    )

                    if patch % 100 == 0:
                        print(f'Writing {patch} of {patches}')

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={'class': feature}
                        )
                    )

                    writer.write(example.SerializeToString())
                    patch += 1
    elif predict and nested:
        print('storing nested predictions')
        folders = tf.io.gfile.glob(os.path.join(bucket, test_folder))

        for folder in folders:
            _, scene_id = folder.split()

            mixer_file = os.path.join(folder, 'mixer.json')
            tfrecord = os.path.join(folder, '00000.tfrecord.gz')
            output_file = os.path.join(folder, scene_id + '-result.tfrecord')

            date = datetime.strptime(scene_id[9:16], '%Y%j')

            uploads.append({
                'mixer': mixer_file,
                'date': f'{date.year:04}-{date.month:02}-{date.day:02}',
                'image': output_file
            })

            with tf.io.gfile.GFile(mixer_file, 'r') as f:
                mixer = json.loads(f.read())
            patches = mixer['totalPatches']

            print(f'building dataset for {tfrecord}')
            dataset = Data.get_dataset(
                patterns=tfrecord, shape=shape, image_bands=image_bands,
                annotation_bands=None, batch_size=1, filters=False,
                shuffle=False, train=False, burn_class=2
            )
            print('done building dataset')

            predictions = model.predict(dataset, steps=patches, verbose=1)

            print(f'Writing results to {output_file}')

            with tf.io.TFRecordWriter(output_file) as writer:
                patch = 1
                for pred in predictions:
                    k = int(overlap / 2)
                    pred = pred[k:-k, k:-k]
                    value = np.argmax(pred, -1).flatten()
                    feature = tf.train.Feature(
                        int64_list=tf.train.Int64List(value=value)
                    )

                    if patch % 100 == 0:
                        print(f'Writing {patch} of {patches}')

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={'class': feature}
                        )
                    )

                    writer.write(example.SerializeToString())
                    patch += 1
    else:
        print('not storing predictions')

    return model, train_dataset, uploads


if __name__ == '__main__':
    params = {
        'bucket': '/home/rylan/school/historicalFiresQC/',
        'data_pattern': 'data',
        'modelA': 'model1',
        'modelB': 'model2',
        'shape': (128, 128),
        'batch_size': 8,
        'epochs': 2,
        'steps_per_epoch': 25,
        'train_model': True,
        'load_model': False,
        'percent_burn_free': 0.5,
        'min_burn_percent': 0.15
    }

    main(**params)
