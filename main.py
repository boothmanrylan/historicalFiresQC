import os
import tensorflow as tf
import data as Data
import model as Model

def main(bucket='boothmanrylan', data_pattern='rylansPicks*.tfrecord.gz',
         model_pattern='rylansPicksModel', shape=(160, 160), batch_size=32,
         learning_rate=1e-4, epochs=100, steps_per_epoch=100,
         train_model=False, load_model=False,
         min_burn_percent=None, percent_burn_free=None):
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
        cache=True,
        shuffle=True,
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

if __name__ == '__main__':
    import visualize as Visualize
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
