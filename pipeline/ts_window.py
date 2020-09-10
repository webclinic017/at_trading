from typing import List, Tuple
import datetime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.python.ops.gen_dataset_ops import MapDataset

from pipeline.ts_sample_data import make_sample_data_simple, make_sample_data_ts


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


class TSWindowGenerator(object):
    def __init__(self,
                 input_width: int,
                 label_width: int,
                 shift: int,
                 train_df: pd.DataFrame,
                 val_df: pd.DataFrame,
                 test_df: pd.DataFrame,
                 feature_filter_columns: List[str] = None,
                 label_columns: List[str] = None,
                 global_datetime_index_column: str = None):
        """
        overall windows --> |-|-|-|-|-|-|
                            |---|           --> input width
                                |-------|   --> shift
                                    |---|   --> label width

        :param global_datetime_index_column: column name for global datetime index, used to plot and debug
        :param feature_filter_columns: this include both the input columns and label columns
        :param input_width:
        :param label_width:
        :param shift:
        :param train_df:
        :param val_df:
        :param test_df:
        :param label_columns:

        """
        # raw data is stored within the object, this might be an issue if data frame gets big
        self.global_datetime_index_column = global_datetime_index_column
        self.feature_filter_columns = feature_filter_columns
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.label_columns = label_columns
        self.label_columns_indices = None

        assert self.input_width > 0 and self.label_width > 0 and self.shift > 0, \
            'input_width/label_width/shift has to be greater than 0'
        assert self.label_width <= self.shift, 'label_width has to be larger or equal to shift'

        if self.feature_filter_columns is not None:
            self.train_df = self.train_df[self.feature_filter_columns]
            self.val_df = self.val_df[self.feature_filter_columns]
            self.test_df = self.test_df[self.feature_filter_columns]
        self.column_indices = {name: i for i, name in enumerate(self.train_df.columns)}
        if self.label_columns is not None:
            assert np.all(
                [x in self.column_indices.keys() for x in self.label_columns]), 'not all the label columns exist'
            self.label_columns_indices = [self.column_indices[x] for x in self.label_columns]

    def __repr__(self) -> str:
        rep_str = f"""total window width={self.input_width + self.shift}|shift={self.shift}|label width =""" + \
                  f"""{self.label_width}|label columns={self.label_columns}"""
        return rep_str

    def split_window(self, features: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """


        :param features:
        :return:
        """
        return split_window(features,
                            self.input_width,
                            self.label_width,
                            self.shift,
                            self.label_columns_indices)

    def make_dataset(self,
                     input_df: pd.DataFrame,
                     sequence_stride: int = 1,
                     shuffle: bool = False,
                     batch_size: int = 32) -> Tuple[MapDataset, MapDataset]:
        """


        :param batch_size:
        :param shuffle:
        :param sequence_stride:
        :param input_df:
        :return:
        usage:
        >>> input_width = 1000
        >>> shift = 1
        >>> label_width = 1
        >>> sequence_stride = 1
        >>> shuffle = False
        >>> batch_size = 32
        >>> train_df = make_sample_data_ts(read_from_bucket=True)
        >>> val_df = None
        >>> test_df = None
        >>> test_window = TSWindowGenerator(
                input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 None,
                 None)
        >>> test_window.make_dataset(train_df)

        (<tf.Tensor: shape=(2, 2, 3), dtype=float32, numpy=
        array([[[2., 4., 6.],
                [3., 5., 8.]],
               [[1., 3., 4.],
                [2., 4., 6.]]], dtype=float32)>, <tf.Tensor: shape=(2, 1, 3), dtype=float32, numpy=
        array([[[ 4.,  9., 13.]],
               [[ 3.,  5.,  8.]]], dtype=float32)>)

        """
        data = np.array(input_df, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width + self.shift,
            sequence_stride=sequence_stride,
            shuffle=shuffle,
            batch_size=batch_size)
        ds = ds.map(self.split_window)

        # deal with datetime index
        if isinstance(input_df.index, pd.DatetimeIndex):
            input_index = pd.DataFrame(input_df.index.map(datetime.datetime.timestamp))

            ds_index = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=input_index,
                targets=None,
                sequence_length=self.input_width + self.shift,
                sequence_stride=sequence_stride,
                shuffle=shuffle,
                batch_size=batch_size)
            ds_index = ds_index.map(self.split_window)
        else:
            ds_index = None

        return ds, ds_index

    @property
    def train(self) -> Tuple[MapDataset, MapDataset]:
        return self.make_dataset(self.train_df)

    @property
    def val(self) -> Tuple[MapDataset, MapDataset]:
        return self.make_dataset(self.val_df)

    @property
    def test(self) -> Tuple[MapDataset, MapDataset]:
        return self.make_dataset(self.test_df)

    @property
    def sample_train_data(self):
        """

        :return:
        usage:
        >>> input_width = 2
        >>> shift = 1
        >>> label_width = 1
        >>> train_df = make_sample_data_simple()
        >>> train_df = pd.concat([train_df] * 2).reset_index(drop=True)
        >>> val_df = None
        >>> test_df = None
        >>> test_window = TSWindowGenerator(
                input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 None,
                 None)
        >>> test_window.sample_train_data
        """
        train_data, train_index = self.train
        result = next(iter(train_data))
        result_index = next(iter(train_index))
        return result, result_index

    def plot_train_data(self,
                        n_batch: int = 1,
                        n_slice: int = 10,
                        column_filters: List[str] = None,
                        model=None):
        """

        :param model:
        :param n_slice:
        :param column_filters:
        :param n_batch:
        usage:
        >>> n_batch = 1
        >>> n_slice = 5
        >>> column_filters = None
        >>> column_per_figure = True
        >>> input_width = 500
        >>> shift = 5
        >>> label_width = 2
        >>> train_df = make_sample_data_ts(read_from_bucket=True)
        >>> val_df = None
        >>> test_df = None
        >>> self = TSWindowGenerator(
                input_width,
                 label_width,
                 shift,
                 train_df,
                 val_df,
                 test_df,
                 None, ['spx'])

        """
        train_data = list(self.train[0].take(n_batch))
        if self.train[1] is not None:
            train_data_index = list(self.train[1].take(n_batch))
        else:
            train_data_index = None

        if column_filters is None:
            plot_columns_dict = self.column_indices
        else:
            plot_columns_dict = {k: self.column_indices[k] for k in column_filters}

        for k, v in iter(plot_columns_dict.items()):
            batch_range = range(n_batch)

            for idx in batch_range:
                train_data_per_iteration = train_data[idx]
                if train_data_index is not None:
                    train_data_index_per_iteration = train_data_index[idx]
                    inputs_index, labels_index = train_data_index_per_iteration
                else:
                    inputs_index = np.arange(self.input_width + self.shift)[slice(0, self.input_width)]
                    labels_index = np.arange(self.input_width + self.shift)[
                        slice(self.input_width + self.shift - self.label_width, None)]
                inputs, labels = train_data_per_iteration

                n_slice = min(len(inputs), n_slice)
                fig, axs = plt.subplots(n_slice, sharex=True)
                ax_dict = dict(zip(list(range(n_slice)), axs))
                fig.autofmt_xdate()
                for slice_id in range(n_slice):
                    ax = ax_dict[slice_id]
                    ax.grid(True)

                    slice_index = inputs_index[slice_id, :, :]
                    slice_label_index = labels_index[slice_id, :, :]
                    if train_data_index is not None:
                        slice_index = np.vectorize(datetime.datetime.fromtimestamp)(
                            tf.reshape(slice_index, slice_index.shape[0]).numpy())

                        slice_label_index = np.vectorize(datetime.datetime.fromtimestamp)(
                            tf.reshape(slice_label_index, slice_label_index.shape[0]).numpy())

                    ax.plot(slice_index, inputs[slice_id, :, v], marker='.', alpha=0.8)
                    if self.label_columns is None or k in self.label_columns:
                        ax.scatter(slice_label_index, labels[slice_id, :, v], alpha=0.8)

                    ax.set_title('feature [{}] - batch [{}] - slice [{}] '.format(k, idx, slice_id))
