import os
import json
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
import data as Data
import model as Model
import assessment as Assessment

pd.options.display.max_columns = 15

valid_annotation_types = ['level_slice', 'bounding_box']
valid_outputs = ['all', 'burn_age', 'burn']
valid_loss_functions = ['basic', 'weighted', 'reference_point', 'no_burn_edge']
valid_bafs = ['scale', 'log', 'sigmoid', None]

# when adding new model parameters that were not previously being tracked
# add them here with their that all previously trained models will have so that
# we can still compare them to newer runs
default_values = {
    'Augment Data': False,
    'Use Previous Classification': False,
    'Normalized Data': False,
    'Minimum Burn Percentage': None,
    'Burn Free Patches': None,
    'Percentage Burn Free': None
}

def main(bucket='boothmanrylan', data_folder='historicalFiresQCInput',
         model_folder='historicalFiresModels', annotation_type='level_slice',
         output='all', shape=(128, 128), kernel=128, batch_size=100,
         stack_image=False, include_previous_burn_age=False,
         include_previous_class=False, burn_age_function='scale',
         learning_rate=1e-4, epochs=100, steps_per_epoch=100,
         train_model=False, load_model=True, loss_function='basic',
         store_predictions=False, augment_data=True, assess_model=False,
         include_tca=False, min_burn_percent=None, percent_burn_free=None):
    # ==========================================================
    # CHECK THAT ARGUMENTS ARE VALID
    # ==========================================================
    bad_arg = 'Invalid choice for argument'
    assert annotation_type in valid_annotation_types, bad_arg
    assert output in valid_outputs, bad_arg
    assert loss_function in valid_loss_functions, bad_arg
    assert burn_age_function in valid_bafs, bad_arg

    if min_burn_percent is not None:
        try:
            assert output in ['all', 'burn']
        except AssertionError as E:
            raise ValueError('Cannot enforce minimum burn percentage') from E
    if percent_burn_free is not None:
        try:
            assert output in ['all', 'burn']
        except AssertionError as E:
            raise ValueError('Cannot filter for patches with no burns') from E

    # =========================================================
    # SET THE PATHS TO THE DATA AND MODELS
    # =========================================================
    data_folder = os.path.join(bucket, data_folder)
    model_folder = os.path.join(bucket, model_folder)

    metadatafile = os.path.join(model_folder, 'models.csv')
    train_pattern = os.path.join(data_folder, 'train*.tfrecord.gz')
    val_pattern = os.path.join(data_folder, 'val*.tfrecord.gz')
    # test_pattern = os.path.join(data_folder, 'test*.tfrecord.gz')

    normalized_data = bool('NormalizedData' in data_folder)

    if include_tca:
        try:
            assert normalized_data
        except AssertionError as E:
            raise ValueError('TCA only exists in normalized data') from E

    # ==========================================================
    # SET THE BANDS TO USE AS OUTPUT FOR THE MODEL
    # ==========================================================
    print('Creating datasets...')

    if normalized_data:
        classes = 3 # None, land, burn
        labels = ['None', 'Land', 'Burn']
        combine = None
    else:
        classes = 5 # none, cloud, water, land, burn
        labels = ['None', 'Cloud', 'Water', 'Land', 'Burn']
        combine = [(5, 4)] # merge new burns and old burn into one class

    if annotation_type == 'level_slice':
        annotation_bands = ['lsliceClass']
    elif annotation_type == 'bounding_box':
        annotation_bands = ['bboxClass']
    else:
        raise ValueError(f'Bad annotation type: {annotation_type}')

    if output == 'burn_age':
        classes = 1 # predicting a continuout output
        labels = None
        if annotation_type == 'level_slice':
            annotation_bands = ['lsliceBurnAge']
        else:
            annotation_bands = ['bboxBurnAge']
    elif output == 'burn':
        labels = ['Not Burnt', 'Burnt']
        classes = 2 # predicting burn vs not burn
        # convert all non burn classes to 0 and all burn classes to 1
        if normalized_data:
            combine = [(1, 0), (2, 1)]
        else:
            combine = [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1)]

    print(f'Using {annotation_bands} as ground truth.')

    # ==========================================================
    # SET THE BANDS TO USE AS INPUT TO THE MODEL
    # ==========================================================
    image_bands = ['B4', 'B5', 'B6', 'B7']
    if include_tca:
        image_bands.append('TCA')

    if stack_image:
        image_bands.extend(['OldB4', 'OldB5', 'OldB6', 'OldB7'])
        if include_tca:
            image_bands.append('TCA')

    if include_previous_burn_age:
        if annotation_type == 'level_slice':
            image_bands.append('prevLSliceBurnAge')
        else:
            image_bands.append('prevBBoxBurnAge')
    if include_previous_class:
        if annotation_type == 'level_slice':
            image_bands.append('prevLSliceClass')
        else:
            image_bands.append('prevBBoxClass')
    channels = len(image_bands)

    print(f'Using {image_bands} as input to model.')
    print(f'Input has {channels} channels. Output has {classes} classes.')

    # ===========================================================
    # BUILD THE DATASETS
    # ===========================================================
    if burn_age_function is None:
        baf = lambda x: x
    elif burn_age_function == 'scale':
        baf = Data.scale_burn_age
    elif burn_age_function  == 'log':
        baf = Data.log_burn_age
    elif burn_age_function == 'sigmoid':
        baf = Data.sigmoid_burn_age
    else:
        raise ValueError(f'Bad burn age function {burn_age_function}')

    # add kernel to shape to include overlap in earth engine patches
    shape = (shape[0] + kernel, shape[1] + kernel)

    if output == 'burn_age':
        train_filter = Data.filter_all_max_burn_age
    elif output == 'burn':
        train_filter = lambda im, annot: Data.filter_no_x(1, im, annot)
    else:
        if normalized_data:
            train_filter = lambda im, annot: Data.filter_no_x(2, im, annot)
        else:
            train_filter = Data.filter_no_burnt

    if min_burn_percent is not None:
        if output == 'all':
            if normalized_data:
                burn_class = 2
            else:
                burn_class = 4
        else:
            burn_class = 1
        train_filter = lambda im, annot: Data.filter_mostly_burnt(
            im, annot, burn_class, min_burn_percent
        )

    if loss_function == 'no_burn_edge': # add burn edge mask to annotationS
        if annotation_type == 'level_slice':
            annotation_bands.append('lsliceBurnEdges')
        else:
            annotation_bands.append('bboxBurnEdges')
    elif loss_function == 'reference_point': # add reference points
        annotation_bands.append('referencePoints')

    train_dataset = Data.get_dataset(
        patterns=train_pattern, shape=shape,
        image_bands=image_bands, annotation_bands=annotation_bands,
        combine=combine, burn_class=burn_class,
        batch_size=batch_size, filters=train_filter, shuffle=True,
        repeat=True, prefetch=True, cache=True,
        burn_age_function=baf, augment=augment_data,
        percent_burn_free=percent_burn_free
    )

    if loss_function == 'no_burn_edge': # remove the burn edge mask
        annotation_bands = annotation_bands[:-1]
    elif loss_function == 'reference_point': # remove reference points
        annotation_bands = annotation_bands[:-1]

    # TODO NOW: out of image burn age doens't make sense 0 === most burned

    val_dataset = Data.get_dataset(
        patterns=val_pattern, shape=shape,
        image_bands=image_bands, annotation_bands=annotation_bands,
        combine=combine, batch_size=batch_size, filters=None,
        shuffle=False, repeat=False, prefetch=True, cache=True,
        burn_age_function=baf, augment=False, burn_class=burn_class
    )

    ref_point_dataset = Data.get_dataset(
        patterns=val_pattern, shape=shape,
        image_bands=image_bands, annotation_bands=['referencePoints'],
        combine=combine, batch_size=1, filters=True, shuffle=False,
        repeat=False, prefetch=True, cache=True, burn_age_function=None,
        augment=False, burn_class=burn_class
    )

    # test_dataset = Data.get_dataset(
    #     patterns=test_pattern, shape=shape,
    #     image_bands=image_bands, annotation_bands=annotation_bands,
    #     combine=combine, batch_size=1, filters=None, shuffle=False,
    #     repeat=False, prefetch=True, cache=True,
    #     burn_age_function=baf
    # )

    print('Done creating datasets.\n')

    # =============================================================
    # SET UP METADATA TO BE LOGGED TODO: replace this with MLMD
    # =============================================================
    print('Setting up metadata to be logged...')

    def pretty(obj):
        if isinstance(obj, str):
            output = obj.replace('_', ' ').title()
        elif obj is None:
            output = 'None'
        else:
            output = obj
        return output

    model_parameters = {
        'Annotation Type': pretty(annotation_type),
        'Loss Function': pretty(loss_function),
        'Stacked Images': pretty(stack_image),
        'Use Previous Burn Age': pretty(include_previous_burn_age),
        'Use Previous Classification': pretty(include_previous_class),
        'Output': pretty(output),
        'Learning Rate': pretty(learning_rate),
        'Burn Age Function': pretty(burn_age_function),
        'Augment Data': augment_data,
        'Normalized Data': pretty(normalized_data),
        'Minimum Burn Percentage': pretty(min_burn_percent),
        'Burn Free Patches': pretty(None),
        'Percentage Burn Free': pretty(percent_burn_free)
    }

    columns = ['Model', 'Date', 'Epochs'] + list(model_parameters.keys())

    current_model = pd.DataFrame(model_parameters, index=[0])

    # open the metadata file if it exists otherwise create a new one
    try:
        metadata = pd.read_csv(metadatafile, index_col=False)
    except FileNotFoundError:
        metadata = pd.DataFrame(columns=columns)

    try:
        all_models = metadata[model_parameters.keys()]
    except KeyError: # new model parameter added
        # add missing parameters with default value for that parameter
        missing = [x for x in model_parameters if x not in metadata.columns]
        print(f'Updating metadata to include new parameters: {missing}')
        for elem in missing:
            metadata[elem] = pretty(default_values[elem])
        all_models = metadata[model_parameters.keys()]

    # check if a model with these same parameters has already been trained
    prev_models = all_models[(all_models.values == current_model.values).all(1)]
    if not prev_models.empty:
        index = prev_models.index[0]
        model_parameters = metadata.to_dict('records')[index]
        metadata = metadata.drop(index)
        model_number = model_parameters['Model']
        print(f'Found model {model_number:05d} with matching parameters.')
        if train_model:
            if load_model:
                model_parameters['Epochs'] += epochs
            else:
                model_parameters['Epochs'] = epochs
    else:
        print('No previous model with the same parameters was found.')
        if metadata.shape[0] > 0:
            model_number = metadata['Model'].max() + 1
        else:
            model_number = metadata.shape[0]
        assert model_number not in metadata['Model']
        model_parameters['Model'] = model_number
        if train_model:
            model_parameters['Epochs'] = epochs

    if train_model:
        model_parameters['Date'] = pd.to_datetime(datetime.now())

    model_parameters = pd.DataFrame(
        model_parameters,
        index=[model_number],
        columns=columns
    )

    print(f'Model parameters:\n{model_parameters}')

    # update metadata file and write out
    metadata = metadata.append(model_parameters)
    metadata.to_csv(metadatafile, index=False)

    print('Done saving metadata.\n')

    # ============================================================
    # BUILD/LOAD THE MODEL
    # ============================================================
    print('Building model...')

    model_path = os.path.join(model_folder, f"{model_number:05d}/")
    model = Model.build_unet_model(
        input_shape=(*shape, channels), classes=classes
    )

    if load_model:
        print(f'Loading model weights from {model_path}...')
        assert not prev_models.empty, "Cannot load weights, no model exists"
        if not train_model:
            # use expect partial to squash unresolved object warnings
            model.load_weights(model_path).expect_partial()
        else:
            model.load_weights(model_path)
        print('Done loading model weights.\n')
    else:
        print('Not loading model weights.')

    if train_model:
        print('Training model...')
        print(f'Saving model weights to {model_path}.')
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, save_weights_only=True,
            save_freq=steps_per_epoch
        )
        callbacks = [checkpoint]

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        if output == 'burn_age':
            base_loss_fn = tf.keras.losses.MAE
            args = {}
        else:
            base_loss_fn = tf.keras.losses.sparse_categorical_crossentropy
            args = {'from_logits': True}

        if loss_function == 'weighted':
            # TODO: allow weights to be passed by user
            if classes == 5:
                weights = [0.1, 10.0, 10.0, 0.1, 10.0]
            elif classes == 2:
                weights = [0.1, 1]
            else:
                raise ValueError('Cannot use weighted loss for burn age')

            loss_fn = Model.weighted_loss(
                base_loss_fn,
                weights,
                **args
            )
        elif loss_function == 'reference_point':
            # TODO: add ability to make this weighted or not
            # TODO: add ability to set alpha and beta here
            loss_fn = Model.reference_point_loss(base_loss_fn, **args)
        elif loss_function == 'no_burn_edge':
            loss_fn = Model.no_burn_edge_loss(base_loss_fn, **args)
        elif loss_function == 'basic':
            loss_fn = Model.basic_loss(base_loss_fn, **args)
        else:
            raise ValueError(f'bad loss function: {loss_function}')

        if output == 'burn_age':
            metrics = ['mse']
        else:
            metrics = ['accuracy']
            # metrics = [Model.MeanIoUFromLogits(classes),
            #            tf.keras.metrics.SparseCategoricalAccuracy()]

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

        model.fit(
            train_dataset, epochs=epochs, verbose=1,
            steps_per_epoch=steps_per_epoch, callbacks=callbacks
        )
        print('Done training model.\n')

    print('Done building model.\n')

    # ================================================================
    # ASSESS THE MODEL
    # ================================================================
    assessment_path = os.path.join(model_path, 'assessment.csv')
    acc_assessment = None
    if train_model and assess_model:
        print('Assessing model performance...')
        acc_assessment = Assessment.assessment(
            model, ref_point_dataset, output, max_burn_age=baf(3650),
            kernel=kernel, labels=labels
        )
        print(f'Saving model assessment to {assessment_path}.')
        acc_assessment.to_csv(assessment_path)

        # TODO: make accuracy by burn age plots

        print('Done assessing model.\n')
    else:
        if load_model:
            print(f'Loading previous assessment from {assessment_path}')
            try:
                acc_assessment = pd.read_csv(assessment_path, index_col=0)
            except FileNotFoundError:
                print(f'Assessment file {assessment_path} missing.')
                if assess_model:
                    print('Redoing model assessment.')
                    acc_assessment = Assessment.assessment(
                        model, ref_point_dataset, output,
                        max_burn_age=baf(3650), kernel=kernel, labels=labels
                    )
                    print(f'Saving model assessment to {assessment_path}')
                    acc_assessment.to_csv(assessment_path)
                    print('Done assessing model.\n')

    # =================================================================
    # UPLOAD PREDICTIONS TO EARTH ENGINE
    # =================================================================
    if store_predictions:
        print(f'Storing predictions in {model_path}')

        mixer_files = tf.io.gfile.glob(os.path.join(data_folder, '*mixer.json'))

        for m in mixer_files:
            with tf.io.gfile.GFile(m, 'r') as f:
                mixer = json.loads(f.read())
            patches = mixer['totalPatches']

            pattern = m.replace('mixer.json', '*.tfrecord.gz')
            tfrecords = tf.io.gfile.glob(pattern)
            tfrecords.sort()

            dataset = Data.get_dataset(
                tfrecords, shape, image_bands, annotation_bands,
                combine=combine, batch_size=1, filters=False, shuffle=False
            )

            predictions = model.predict(dataset, steps=patches, verbose=1)

            filename = m.replace('-mixer.json', '-results.tfrecord')
            filename = filename.replace(data_folder, '')
            if filename[0] == '/': # remove erroneous /
                filename = filename[1:]
            output_file = os.path.join(model_path, filename)

            print(f'Writing results for {m} to {output_file}')

            with tf.io.TFRecordWriter(output_file) as writer:
                patch = 1
                for pred in predictions:
                    k = int(kernel / 2)
                    pred = pred[k:-k, k:-k]
                    if output != 'burn_age':
                        value = np.argmax(pred, -1).flatten()
                        feature = tf.train.Feature(
                            int64_list=tf.train.Int64List(value=value)
                        )
                    else:
                        value = pred.flatten()
                        feature = tf.train.Feature(
                            float_list=tf.train.FloatList(value=value)
                        )

                    if patch % 100 == 0:
                        print(f'Writing {patch} of {patches}')

                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={output: feature}
                        )
                    )

                    writer.write(example.SerializeToString())
                    patch += 1

        print('Done storing predictions.\n')

    print('Main completed.')

    output = {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'ref_point_dataset': ref_point_dataset,
        'model': model,
        'assessment': acc_assessment,
        'data_folder': data_folder,
        'model_folder': model_path,
        'model_number': model_number,
        'burn_age_function': baf
    }

    return output

if __name__ == '__main__':
    import visualize as Visualize
    params = {
        'bucket': '/home/rylan/school/historicalFiresQC/',
        'data_folder': 'data',
        'model_folder': 'models',
        'shape': (128, 128),
        'kernel': 32,
        'batch_size': 8,
        'epochs': 2,
        'steps_per_epoch': 25,
        'train_model': True,
        'load_model': False,
        'store_predictions': False,
        'loss_function': 'basic',
        'output': 'all',
        'augment_data': True,
        'assess_model': False,
        'stack_image': False,
        'include_previous_burn_age': False,
        'include_previous_class': False,
        'percent_burn_free': 0.5,
        'min_burn_percent': 0.15
    }
    test_result = main(**params)
    if params['output'] == 'burn_age':
        max_annot = test_result['burn_age_function'](3650)
    elif params['output'] == 'burn':
        max_annot = 1
    else:
        max_annot = None

    Visualize.visualize(
        test_result['train_dataset'], test_result['model'], num=20,
        stacked_image=params['stack_image'],
        include_prev_burn_age=params['include_previous_burn_age'],
        include_prev_class=params['include_previous_class'],
        max_annot=max_annot, max_burn_age=test_result['burn_age_function'](3650)
    )
