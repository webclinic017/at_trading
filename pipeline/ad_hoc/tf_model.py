from abc import ABC

import numpy as np
import tensorflow as tf
import os
import pandas as pd
import datetime
from matplotlib import pyplot as plt
import seaborn as sns


class WindowGenerator(object):
    def __init__(self,
                 input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 label_columns=None):
        """

        :param input_width:
        :param label_width:
        :param shift:
        :param train_df:
        :param val_df:
        :param test_df:
        :param label_columns:
        """
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


MAX_EPOCHS = 100


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train,
                        epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def gen_test_data():
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)
    df = pd.read_csv(csv_path)
    # slice [start:stop:step], starting from index 5 take every 6th record.
    df = df[5::6]
    df = df.reset_index(drop=True)
    date_time_ts = pd.to_datetime(df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    wv = df['wv (m/s)']
    bad_wv = wv == -9999.0
    wv[bad_wv] = 0.0

    max_wv = df['max. wv (m/s)']
    bad_max_wv = max_wv == -9999.0
    max_wv[bad_max_wv] = 0.0

    # The above inplace edits are reflected in the DataFrame
    df['wv (m/s)'].min()

    wv = df.pop('wv (m/s)')
    max_wv = df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    df['Wx'] = wv * np.cos(wd_rad)
    df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    df['max Wx'] = max_wv * np.cos(wd_rad)
    df['max Wy'] = max_wv * np.sin(wd_rad)

    timestamp_s = date_time_ts.map(datetime.datetime.timestamp)
    day = 24 * 60 * 60
    year = (365.2425) * day

    df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df


class BaselineModel(tf.keras.Model, ABC):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def test_func():
    train_df, val_df, test_df = gen_test_data()
    w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
                         label_columns=['T (degC)'],
                         train_df=train_df,
                         val_df=val_df,
                         test_df=test_df
                         )
    # Stack three slices, the length of the total window:
    example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                               np.array(train_df[100:100 + w2.total_window_size]),
                               np.array(train_df[200:200 + w2.total_window_size])])

    example_inputs, example_labels = w2.split_window(example_window)
    # w2.example = example_inputs, example_labels
    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')
    for example_inputs, example_labels in w2.train.take(2):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    column_indices = {name: i for i, name in enumerate(train_df.columns)}

    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1,
        label_columns=['T (degC)'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    baseline = BaselineModel(label_index=column_indices['T (degC)'])

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1,
        label_columns=['T (degC)'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )
    wide_window.plot(baseline)

    CONV_WIDTH = 3
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=['T (degC)'],
        train_df=train_df,
        val_df=val_df,
        test_df=test_df
    )

    conv_window.plot()

    multi_step_dense = tf.keras.Sequential([
        # Shape: (time, features) => (time*features)
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
        # Add back the time dimension.
        # Shape: (outputs) => (1, outputs)
        tf.keras.layers.Reshape([1, -1]),
    ])
    multi_step_dense(conv_window.example[0])

    history = compile_and_fit(multi_step_dense, conv_window)
    val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
    performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
    conv_window.plot(multi_step_dense)

    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])
    print("Conv model on `conv_window`")
    print('Input shape:', conv_window.example[0].shape)
    print('Output shape:', conv_model(conv_window.example[0]).shape)

    history = compile_and_fit(conv_model, conv_window)
    val_performance['Conv'] = conv_model.evaluate(conv_window.val)
    performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])
    history = compile_and_fit(lstm_model, wide_window)
    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
