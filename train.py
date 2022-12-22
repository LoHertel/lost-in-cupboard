import argparse
import copy
import gc
import os
import json 
import math 
from datetime import datetime

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import PIL
# supress info, warning and error messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.applications import EfficientNetV2L
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import disable_interactive_logging, enable_interactive_logging, is_interactive_logging_enabled

from src.kitchenware_helper import (
    get_latest_version,
    load_history_version,
    track_experiment,
)



def train_efficientnet_v2_l(df: pd.DataFrame, epochs: int, params: dict, verbose: bool = False) -> pd.DataFrame:
    """Trains a model based on EfficientNetV2L.
    
    :param pd.DataFrame df: DataFrame with image data for training.
    :param int epochs: Number of epochs the model should train.
    :param dict params: Hyper-parameters the model training should bebased on.

    Returns a DataFrame with the training history.
    """

    # control printing
    if verbose:
        enable_interactive_logging()
    else:
        disable_interactive_logging()

    # new version / id number for experiment tracking
    version: int = get_latest_version() + 1
    
    # train / val split
    # split into 80% train and 20% validation
    #val_cutoff = int(len(df) * 0.8)
    #df_train = df[:val_cutoff]
    #df_val = df[val_cutoff:]
    

    # image augmentation
    image_size = params.get('image_size')

    datagen_augmentation = ImageDataGenerator(
        validation_split = 0.2,
        **params.get('augmentation', {})
    )

    train_generator = datagen_augmentation.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='label',
        target_size=image_size,
        batch_size=32,
        subset='training'
    )

    val_generator = datagen_augmentation.flow_from_dataframe(
        df,
        x_col='filename',
        y_col='label',
        target_size=image_size,
        batch_size=32,
        subset='validation'
    )

    labels = list(train_generator.class_indices.keys())

    # base model
    input_shape = image_size + (3, )

    base_model = EfficientNetV2L(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape, # default 299, 299, 3
        pooling=params.get('model').get('pooling', None),
        include_preprocessing=True
    )
    base_model.trainable = False
    
    # dense layers
    inputs = Input(shape=input_shape)
    layers = base_model(inputs, training=False)

    if params.get('model').get('dense', 0) > 0:
        layers = Dense(params.get('model').get('dense', 0), activation='relu')(layers)
    
    if params.get('model').get('bn', False):
        layers = BatchNormalization()(layers)
    
    if params.get('model').get('dropout', 0) > 0:
        layers = Dropout(params.get('model').get('dropout', 0))(layers)

    outputs = Dense(6)(layers)

    model = Model(inputs, outputs)

    if is_interactive_logging_enabled():
        print(model.summary())

    # compile model
    optimizer = Adam(learning_rate=params.get('training').get('lr', 0.001))
    loss = CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])


    # model training
    checkpoint = ModelCheckpoint(
        'models/efficientnetv2l_v%d_{epoch:02d}_{val_accuracy:.4f}.h5' % (version),
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        initial_value_threshold=0.9 # save only models which have more than 90% validation accuracy.
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint],
        #initial_epoch = 0
    )

    # free memory immediatly
    del model
    gc.collect()

    track_experiment(
        history, 
        architecture='EfficientNetV2L',
        version=version,
        hyper_params=params,
        labels=labels
    )

    # reenable Keras output
    if verbose is False:
        enable_interactive_logging()

    return version


if __name__ == '__main__':

    # parse arguments
    parser = argparse.ArgumentParser(description="Trains a Keras model für the Kitchenware competition.",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("epochs", nargs='?', default=10, help="Number of epochs for training", type=int)
    parser.add_argument("train_data", nargs='?', default='data/train.csv', help="Source location of the model's DataFrame with labels forthe image")
    args = parser.parse_args()
    config = vars(args)


    # final training parameter 
    final_params = {
        'image_size': (150, 150),
        
        'augmentation': {
            'horizontal_flip': True, 
            'vertical_flip': True,
            'rotation_range': 90, 
            'fill_mode': 'nearest',
        },
        
        'model': {
            'pooling': 'max',
            'dense': 64,
            'bn': True,
            'dropout': 0.5
        },

        'training': {
            'lr': 0.0005
        }  
    }

    # load labels
    df_train_full = pd.read_csv(config['train_data'], dtype={'Id': str})
    df_train_full['filename'] = 'data/images/' + df_train_full['Id'] + '.jpg'


    # training
    training_id = train_efficientnet_v2_l(df_train_full, epochs=config['epochs'], params=final_params, verbose=True)

    # report on training
    df_history = load_history_version(training_id)

    print('Training was finished.')
    print('Training-ID:', training_id)
    print('max. training accuracy:', df_history['accuracy'].max())
    print('max. validation accuracy:', df_history['val_accuracy'].max())