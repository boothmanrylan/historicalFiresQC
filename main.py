import os
import json
import subprocess
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

def main(bucket='boothmanrylan', data_folder='historicalFiresQCInput',
         model_folder='historicalFiresModels', annotation_type='level_slice',
         output='all', shape=(128, 128), kernel=128, batch_size=100,
         stack_image=False, include_previous_burn_age=False,
         burn_age_function='scale', learning_rate=1e-4, epochs=100,
         steps_per_epoch=100, train_model=False, load_model=True,
         loss_function='basic', store_predictions=False, augment_data=True):
    # ==========================================================
    # CHECK THAT ARGUMENTS ARE VALID
    # ==========================================================
    bad_arg = 'Invalid choice for argument'
    assert annotation_type in valid_annotation_types, bad_arg
    assert output in valid_outputs, bad_arg
    assert loss_function in valid_loss_functions, bad_arg
    assert burn_age_function in valid_bafs, bad_arg

    # =========================================================
    # SET THE PATHS TO THE DATA AND MODELS
    # =========================================================
    data_folder = os.path.join(bucket, data_folder)
    model_folder = os.path.join(bucket, model_folder)

    metadatafile = os.path.join(model_folder, 'models.csv')
    train_pattern = os.path.join(data_folder, 'train*.tfrecord.gz')
    val_pattern = os.path.join(data_folder, 'val*.tfrecord.gz')
    # test_pattern = os.path.join(data_folder, 'test*.tfrecord.gz')

    # ==========================================================
    # SET THE BANDS TO USE AS OUTPUT FOR THE MODEL
    # ==========================================================
    print('Creating datasets...')

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
        combine = [(1, 0), (2, 0), (3, 0), (4, 1), (5, 1)]

    print(f'Using {annotation_bands} as ground truth.')

    # ==========================================================
    # SET THE BANDS TO USE AS INPUT TO THE MODEL
    # ==========================================================
    image_bands = ['B4', 'B5', 'B6', 'B7']

    if stack_image:
        image_bands.extend(['OldB4', 'OldB5', 'OldB6', 'OldB7'])

    if include_previous_burn_age:
        if annotation_type == 'level_slice':
            image_bands.append('prevLSliceBurnAge')
        else:
            image_bands.append('prevBBoxBurnAge')
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
        train_filter = lambda image, annot: Data.filter_no_x(1, image, annot)
    else:
        train_filter = Data.filter_no_burnt

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
        combine=combine,
        batch_size=batch_size, filters=train_filter, shuffle=True,
        repeat=True, prefetch=True, cache=True,
        burn_age_function=baf, augment=augment_data
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
        burn_age_function=baf, augment=False
    )

    ref_point_dataset = Data.get_dataset(
        patterns=val_pattern, shape=shape,
        image_bands=image_bands, annotation_bands=['referencePoints'],
        combine=combine, batch_size=1, filters=True, shuffle=False,
        repeat=False, prefetch=True, cache=True, burn_age_function=None,
        augment=False
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
        else:
            output = obj
        return output

    model_parameters = {
        'Annotation Type': pretty(annotation_type),
        'Loss Function': pretty(loss_function),
        'Stacked Images': pretty(stack_image),
        'Use Previous Burn Age': pretty(include_previous_burn_age),
        'Output': pretty(output),
        'Learning Rate': pretty(learning_rate),
        'Burn Age Function': pretty(burn_age_function),
        'Augment Data': augment_data
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
        # add missing parameters with None to all previous models
        missing = [x for x in model_parameters.keys()
                   if x not in metadata.columns]
        print(f'Updating metadata to include new parameters: {missing}')
        for elem in missing:
            metadata[elem] = None
        all_models = metadata[model_parameters.keys()]

    # check if a model with these same parameters has already been trained
    prev_models = all_models[(all_models.values == current_model.values).all(1)]
    if not prev_models.empty:
        index = prev_models.index[0]
        model_number = prev_models.index[0]
        print(f'Found model {model_number:05d} with matching parameters.')
        model_parameters = metadata.to_dict('records')[index]
        metadata = metadata.drop(index)
        model_number = model_parameters['Model']
        if train_model:
            if load_model:
                model_parameters['Epochs'] += epochs
            else:
                model_parameters['Epochs'] = epochs
    else:
        print('No previous model with the same parameters was found.')
        model_number = metadata.shape[0]
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
    if train_model:
        print('Assessing model performance...')
        if output == 'burn_age':
            acc_assessment = Assessment.burn_age_accuracy_assessment(
                model, ref_point_dataset, baf(3650), kernel
            )
        else:
            acc_assessment = Assessment.classification_accuracy_assessment(
                model, ref_point_dataset, labels
            )

        print(f'Saving model assessment to {assessment_path}.')
        acc_assessment.to_csv(assessment_path)

        # TODO: make accuracy by burn age plots

        print('Done assessing model.\n')
    else:
        if load_model:
            print(f'Loading previous assessment from {assessment_path}')
            acc_assessment = pd.read_csv(assessment_path, index_col=0)

    # =================================================================
    # UPLOAD PREDICTIONS TO EARTH ENGINE
    # =================================================================
    if store_predictions:
        print(f'Storing predictions in {model_path}')

        try:
            result = subprocess.run(
                ['gsutil', 'ls', data_folder],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                check=True
            )
        except subprocess.CalledProcessError as E:
            print(E.stderr)
            raise E

        all_files = result.stdout.split('\n')
        mixer_files = [x for x in all_files if '.json' in x]

        for m in mixer_files:
            try:
                result = subprocess.run(
                    ['gsutil', 'cat', m],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    check=True
                )
            except subprocess.CalledProcessError as E:
                print(E.stderr)
                raise E

            mixer = json.loads(result.stdout)
            patches = mixer['totalPatches']

            tfrecords = [
                x for x in all_files if m.replace('mixer.json', '') in x
            ]
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
        'model_number': model_number
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
        'steps_per_epoch': 1,
        'train_model': True,
        'load_model': False,
        'store_predictions': False,
        'loss_function': 'no_burn_edge',
        'output': 'burn',
        'augment_data': False
    }
    output = main(**params)
    Visualize.visualize(output['train_dataset'], num=20)

