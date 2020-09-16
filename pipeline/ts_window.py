from typing import List, Tuple, Dict
import datetime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from numpy import ma
from tensorflow.python.ops.gen_dataset_ops import MapDataset

from pipeline.ts_feature import split_data_frame
from pipeline.ts_sample_data import make_sample_data_simple, make_sample_data_ts, make_sample_data_random_ts


def split_window(input_features: tf.Tensor,
                 input_width: int,
                 label_width: int,
                 shift: int,
                 label_column_indices: List[int] = None) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    given a stacked tensor, each represent a subset of timeseries data called batch,
    split the data to inputs(time series blocks to train)/labels (time block to predict)
    e.g. say if we have a tensor which has a shape of [2, 3, 2]
    which means there are two batches and 3 inputs and 2 features

    array([[[1, 3],
        [2, 4],
        [3, 5]],
       [[2, 4],
        [3, 5],
        [4, 9]]])

    say we want to break this tensor such that for each batch, first two periods are inputs and 3rd
    period is the label, in terms of parameters it means input_width = 2, label_width = 1, shift = 1 then we will get

    inputs:
    array([[[1, 3],
        [2, 4]],
       [[2, 4],
        [3, 5]]])

    labels:
    ray([[[3, 5]],
       [[4, 9]]])

    :param input_features:
    :param input_width:
    :param label_width:
    :param shift:
    :param label_column_indices:
    :return:
    usage:

    >>> input_features = tf.stack([
        np.array(
        [[1, 3], [2, 4], [3, 5]]
        ),
        np.array(
        [[2, 4], [3, 5], [4, 9]]
        )
        ])
    >>> input_width = 2
    >>> label_width = 1
    >>> shift=1
    >>> label_indices = None
    >>> split_window(input_features,
                 input_width = input_width,
                 label_width = label_width,
                 shift = shift,
                 label_indices = None)

    (<tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
     array([[[1, 3],
             [2, 4]],

            [[2, 4],
             [3, 5]]], dtype=int32)>,
     <tf.Tensor: shape=(2, 1, 2), dtype=int32, numpy=
     array([[[3, 5]],

            [[4, 9]]], dtype=int32)>)

    >>> input_width = 1
    >>> label_width = 2
    >>> shift=2
    >>> label_indices = None
    >>> split_window(input_features,
                 input_width = input_width,
                 label_width = label_width,
                 shift = shift,
                 label_indices = None)

    (<tf.Tensor: shape=(2, 1, 2), dtype=int32, numpy=
     array([[[1, 3]],

            [[2, 4]]], dtype=int32)>,
     <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
     array([[[2, 4],
             [3, 5]],

            [[3, 5],
             [4, 9]]], dtype=int32)>)


    """
    # couple of assertions
    assert input_width > 0 and label_width > 0 and shift > 0, \
        'input_width/label_width/shift has to be greater than 0'
    assert label_width <= shift, 'label_width has to be larger or equal to shift'

    input_slice = slice(0, input_width)
    label_start = input_width + shift - label_width
    label_slice = slice(label_start, None)
    inputs = input_features[:, input_slice, :]
    labels = input_features[:, label_slice, :]
    if label_column_indices is not None:
        filtered_label_list = [labels[:, :, idx] for idx in label_column_indices]
        labels = tf.stack(filtered_label_list, axis=-1)

    # set shape manually since slicing does not preserve dimensions
    inputs.set_shape([None, input_width, None])
    labels.set_shape([None, label_width, None])
    return inputs, labels


def convert_map_dataset_to_data_frame(input_map: MapDataset,
                                      input_index: MapDataset,
                                      use_datetime_index: bool = True,
                                      max_n_items: int = None) -> List[Dict[str, pd.DataFrame]]:
    """
    input_map is in the format of
    list of matrix in the shape (batch, rows, columns)

    :param max_n_items:
    :param use_datetime_index:
    :param input_map:
    :param input_index:

    usage:
    >>> input_width = 10
    >>> label_width = 2
    >>> shift = 2
    >>> self = TSWindowGenerator(
            input_width = input_width,
            label_width = label_width,
            shift = shift,
    )
    >>> input_df = make_sample_data_random_ts(50, 3)
    >>> input_map, input_index = self.split_convert_to_data_set(input_df)
    >>> use_datetime_index=True
    """
    batch_result_list = []
    for map_item, label_item in zip(input_map.as_numpy_iterator(), input_index.as_numpy_iterator()):
        input_data, label_data = map_item
        input_data_index, label_data_index = label_item
        for idx in range(input_data.shape[0]):
            if use_datetime_index:
                d_index = np.vectorize(datetime.datetime.fromtimestamp)(input_data_index[idx, :].flatten())
                l_index = np.vectorize(datetime.datetime.fromtimestamp)(label_data_index[idx, :].flatten())

            else:
                d_index = input_data_index[idx, :].flatten()
                l_index = label_data_index[idx, :].flatten()
            input_data_df = pd.DataFrame(input_data[idx, :, :],
                                         index=d_index)
            label_data_df = pd.DataFrame(label_data[idx, :, :],
                                         index=l_index)
            batch_result_list.append({
                'input_data': input_data_df,
                'label_data': label_data_df})
            if max_n_items is not None and len(batch_result_list) == max_n_items:
                return batch_result_list

    return batch_result_list


class TSWindowGenerator(object):
    def __init__(self,
                 input_width: int,
                 label_width: int,
                 shift: int,
                 sequence_stride: int = 1,
                 batch_size: int = 32):
        """

        overall windows --> |-|-|-|-|-|-|

        batch 1 --> |-|-|-|-|
                      |-|-|-|-|
                        |-|-|-|-|

        batch 2 -->       |-|-|-|-|
                            |-|-|-|-|
                              |-|-|-|-|

        for each dataset in the batch
                            |-|-|-|-|-|
                            |---|         --> input width
                                |-----|   --> shift
                                    |-|   --> label width
        :param input_width:
        :param label_width:
        :param shift:
        :param sequence_stride:
        :param batch_size:
        usage:
        >>> input_width = 10
        >>> label_width = 2
        >>> shift = 2
        >>> self = TSWindowGenerator(
                input_width = input_width,
                label_width = label_width,
                shift = shift,
        )

        """
        self.batch_size = batch_size
        self.sequence_stride = sequence_stride
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        assert self.input_width > 0 and self.label_width > 0 and self.shift > 0, \
            'input_width/label_width/shift has to be greater than 0'
        assert self.label_width <= self.shift, 'label_width has to be larger or equal to shift'

    def __repr__(self) -> str:
        rep_str = f"""input_width:{self.input_width}|label_width:{self.label_width}|""" + \
                  f"""sequence_stride:{self.sequence_stride}|bath_size:{self.batch_size}|"""
        return rep_str

    def split_convert_to_data_set(self, input_df: pd.DataFrame,
                                  label_column_indices: List[int] = None) -> Tuple[MapDataset, MapDataset]:
        """

        :param input_df:
        :param label_column_indices:
        :return:
        usage:

        >>> input_df = make_sample_data_random_ts(20, 3, random_seed=0)
        >>> label_column_list = [0]


        """
        data = np.array(input_df, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width + self.shift,
            sequence_stride=self.sequence_stride,
            shuffle=False,
            batch_size=self.batch_size)
        ds = ds.map(lambda x: split_window(
            x,
            self.input_width,
            self.label_width,
            self.shift,
            label_column_indices
        ))

        # also generate index as well
        df_index = input_df.index
        if isinstance(input_df.index, pd.DatetimeIndex):
            df_index = df_index.map(datetime.datetime.timestamp)

        ds_index = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=df_index.values.reshape(len(df_index), 1),
            targets=None,
            sequence_length=self.input_width + self.shift,
            sequence_stride=self.sequence_stride,
            shuffle=False,
            batch_size=self.batch_size)
        ds_index = ds_index.map(lambda x: split_window(
            x,
            self.input_width,
            self.label_width,
            self.shift,
            None
        ))
        return ds, ds_index


class TSDataStore(object):
    def __init__(self,
                 window_generator: TSWindowGenerator,
                 in_df: pd.DataFrame,
                 split_dict: Dict = None,
                 label_columns: List[str] = None):
        self.window_generator = window_generator
        self.use_datetime_index = isinstance(in_df.index, pd.DatetimeIndex)

        result_dict = split_data_frame(in_df, split_dict=split_dict)
        self.train_df = result_dict['train']
        self.val_df = result_dict['val']
        self.test_df = result_dict['test']

        self.global_column_index_dict = dict(zip(
            in_df.columns.to_list(),
            list(range(len(in_df.columns)))
        ))

        if label_columns is not None:
            self.label_columns = label_columns
            self.label_column_indices = [self.global_column_index_dict[x] for x in self.label_columns]
        else:
            self.label_columns = in_df.columns.to_list()
            self.label_column_indices = None

        self.model = None
        self.train_prediction_df = None
        self.val_prediction_df = None
        self.test_prediction_df = None

    @property
    def train(self) -> Tuple[MapDataset, MapDataset]:
        return self.window_generator.split_convert_to_data_set(self.train_df)

    @property
    def val(self) -> Tuple[MapDataset, MapDataset]:
        return self.window_generator.split_convert_to_data_set(self.val_df)

    @property
    def test(self) -> Tuple[MapDataset, MapDataset]:
        return self.window_generator.split_convert_to_data_set(self.test_df)

    def compile_and_fit(self,
                        model,
                        patience=2,
                        max_epochs=20):
        """

        :param model:
        :param patience:
        :param max_epochs:
        :return:
        usage:
        >>> multi_lstm_model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(3,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([1, 3])
        ])
        >>> self.compile_and_fit(model)
        """
        self.model = model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                                                          patience=patience,
                                                          mode='min')

        self.model.compile(loss=tf.losses.MeanSquaredError(),
                           optimizer=tf.optimizers.Adam(),
                           metrics=[tf.metrics.MeanAbsoluteError()])

        history = self.model.fit(self.train[0],
                                 epochs=max_epochs,
                                 validation_data=self.val[0],
                                 callbacks=[early_stopping])
        return history

    def populate_predictions(self, mode: List[str] = None):
        """

        :param mode:
        usage:

        >>> model = tf.keras.Sequential([
        # Shape [batch, time, features] => [batch, lstm_units]
        # Adding more `lstm_units` just overfits more quickly.
        tf.keras.layers.LSTM(32, return_sequences=False),
        # Shape => [batch, out_steps*features]
        tf.keras.layers.Dense(3,
                              kernel_initializer=tf.initializers.zeros),
        # Shape => [batch, out_steps, features]
        # tf.keras.layers.Reshape([1, 3])
        ])
        >>> input_width = 10
        >>> label_width = 1
        >>> shift = 1
        >>> window_generator = TSWindowGenerator(
                input_width = input_width,
                label_width = label_width,
                shift = shift,
        )
        >>> in_df = make_sample_data_random_ts(200, 3, random_seed=0)
        >>> self = TSDataStore(window_generator=window_generator,
            in_df=in_df,
            label_columns = [1, 2]
        )
        >>> self.compile_and_fit(model)
        >>> mode = None

        """
        assert self.model is not None, 'please run compile_and_fit first'
        if mode is None:
            populate_list = ['train', 'val', 'test']
        else:
            populate_list = mode

        label_column_indices_map = dict(zip(self.label_columns, self.label_column_indices))
        for populate_item in populate_list:
            data, data_index = getattr(self, populate_item)
            df_list = []
            for map_item, label_item in zip(data.as_numpy_iterator(), data_index.as_numpy_iterator()):
                input_data, label_data = map_item
                input_data_index, label_data_index = label_item
                predictions = self.model(input_data)
                label_dict = {}
                for label_c in self.label_columns:
                    label_dict[label_c] = pd.DataFrame({
                        'prediction': predictions[:, label_column_indices_map[label_c]],
                        'actual': label_data[:, :, label_column_indices_map[label_c]].flatten()
                    }, index=np.vectorize(datetime.datetime.fromtimestamp)(
                        label_data_index.flatten()))
                df_list.append(
                    pd.concat(label_dict, axis=1)
                )
            setattr(self, '{}_prediction_df'.format(populate_item), pd.concat(df_list))

    # plotting functions to visualize the data stored
    def plot_sample_data(self, max_n_items=None) -> List[Figure]:
        """

        :param max_n_items:
        usage:
        >>> input_width = 10
        >>> label_width = 2
        >>> shift = 2
        >>> window_generator = TSWindowGenerator(
                input_width = input_width,
                label_width = label_width,
                shift = shift,
        )
        >>> in_df = make_sample_data_random_ts(100, 3, random_seed=0)
        >>> self = TSDataStore(window_generator=window_generator,
            in_df=in_df
        )
        """
        input_map, input_index = self.train
        df_list = convert_map_dataset_to_data_frame(input_map, input_index, max_n_items=max_n_items)
        fig_list = []
        for df_item in df_list:
            input_data = df_item['input_data']
            label_data = df_item['label_data']
            fig, ax_list = plt.subplots(len(input_data.columns), sharex=True)
            fig.autofmt_xdate()
            ax_dict = dict(zip(input_data.columns, ax_list))
            for col in input_data.columns:
                ax = ax_dict[col]
                ax.grid(True)
                ax.plot(input_data.index, input_data[col].values, marker='.', alpha=0.8, label=col)

                if col in label_data.columns:
                    ax.scatter(label_data.index, label_data[col].values, alpha=0.8, color='grey',
                               label='label')
                ax.legend(loc='upper right', frameon=True)
            fig_list.append(fig)
        return fig_list
