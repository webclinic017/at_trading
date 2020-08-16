import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow_docs as tfdocs
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow_docs.modeling import EpochDots
from tensorflow_docs.plots import HistoryPlotter


def load_data():
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
    # convert to categorical type
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset


def inspect_data(input_df):
    """

    :param input_df:
    usage:
    >>> input_df, test_data = load_data()
    """
    sns.pairplot(input_df[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")


def build_model(input_shape):
    model = keras.Sequential([
        # [len(train_dataset.keys())]
        layers.Dense(64, activation='relu', input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def run_model():
    train_df, test_df = load_data()
    train_label_col = 'MPG'
    train_label = train_df.pop(train_label_col)
    test_label = test_df.pop(train_label_col)
    train_stats = train_df.describe().transpose()

    def normalize_zscore(x):
        return (x - train_stats['mean']) / train_stats['std']

    train_df_norm = normalize_zscore(train_df)
    test_df_norm = normalize_zscore(test_df)
    model = build_model([len(train_df.keys())])
    EPOCHS = 1000

    history = model.fit(
        train_df_norm, train_label,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[EpochDots()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plotter = HistoryPlotter(smoothing_std=2)

    plotter.plot({'Basic': history}, metric="mae")
    plt.ylim([0, 10])
    plt.ylabel('MAE [MPG]')
    model.predict(test_df_norm)

    test_predictions = model.predict(test_df_norm).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_label, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
