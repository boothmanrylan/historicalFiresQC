import json
import os
import tensorflow as tf
import numpy as np
import data as Data
import model as Model

def main(bucket='boothmanrylan', data_pattern='rylansPicks*.tfrecord.gz',
         model_pattern='rylansPicksModel', shape=(160, 160), batch_size=32,
         learning_rate=1e-4, epochs=100, steps_per_epoch=100,
         train_model=False, load_model=False,
         min_burn_percent=None, percent_burn_free=None, predict=False,
         test_folder='historicalFiresQCMaskedData',
         predictions_folder='rylansPicks'):
    image_bands = ['B4', 'B5', 'B6', 'B7', 'TCA', 'bai']
    annotation_bands = ['class']

    train_filter = lambda x, y: Data.filter_mostly_burnt(x, y, 2, min_burn_percent)
    if min_burn_percent == 0:
        train_filter = None

    dataset = Data.get_dataset(
        patterns=os.path.join(bucket, data_pattern),
        shape=shape,
        image_bands=image_bands,
        annotation_bands=annotation_bands,
        batch_size=batch_size,
        filters=train_filter,
        cache=False,
        shuffle=False,
        repeat=True,
        prefetch=True,
        augment=False,
        percent_burn_free=percent_burn_free,
        burn_class=2,
    )

    model_path = os.path.join(bucket, model_pattern)

    channels = len(image_bands)
    classes = 3 # none, not-burn, burn

    model = Model.build_unet_model(
        input_shape=(*shape, channels), classes=classes
    )

    if load_model:
        model.load_weights(model_path).expect_partial()

    if train_model:
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, save_weights_only=True,
            save_freq=steps_per_epoch
        )
        callbacks = [checkpoint]

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = ['accuracy']
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.fit(
            dataset, epochs=epochs, verbose=1,
            steps_per_epoch=steps_per_epoch, callbacks=callbacks
        )

    if predict:
        mixer_files = tf.io.gfile.glob(os.path.join(test_folder, '*mixer.json'))

        for m in mixer_files:
            with tf.io.gfile.GFile(m, 'r') as f:
                mixer = json.loads(f.read())
            patches = mixer['totalPatches']

            pattern = m.replace('mixer.json', '*.tfrecord.gz')
            tfrecords = tf.io.gfile.glob(pattern)
            tfrecords.sort()

            dataset = Data.get_dataset(
                tfrecords, shape, image_bands, annotation_bands,
                batch_size=1, filters=False, shuffle=False, train=False
            )

            predictions = model.predict(dataset, steps=patches, verbose=1)

            filename = m.replace('-mixer.json', '-results.tfrecord')
            filename = filename.replace(test_folder, '')
            if filename[0] == '/': # remove erroneous /
                filename = filename[1:]
            output_file = os.path.join(predictions_folder, filename)

            print(f'Writing results for {m} to {output_file}')

            with tf.io.TFRecordWriter(output_file) as writer:
                patch = 1
                for pred in predictions:
                    k = int(128 / 2)
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


if __name__ == '__main__':
    params = {
        'bucket': '/home/rylan/school/historicalFiresQC/',
        'data_pattern': 'data',
        'model_pattern': 'models',
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
