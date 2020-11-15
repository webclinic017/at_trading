import tensorflow as tf
import pandas as pd
import numpy as np

# similar to pandas rolling function, but works on a numpy matrix
from pipeline.ts_sample_data import make_sample_data_random_ts


def test_tf_dataset():
    # TODO: move this to ts_sample_data
    df = make_sample_data_random_ts(5, 3, random_seed=0)

    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=df.values,
        targets=None,
        sequence_length=3,
        sequence_stride=1,
        shuffle=False,
        batch_size=1)

    # example data looks like the following
    #                   0         1         2
    # 2020-09-08  0.548814  0.715189  0.602763
    # 2020-09-09  0.544883  0.423655  0.645894
    # 2020-09-10  0.437587  0.891773  0.963663
    # 2020-09-11  0.383442  0.791725  0.528895
    # 2020-09-12  0.568045  0.925597  0.071036

    # timeseries dataset is like the following
    # [ < tf.Tensor: shape = (1, 3, 3), dtype = float64, numpy =
    # array([[[0.5488135, 0.71518937, 0.60276338],
    #         [0.54488318, 0.4236548, 0.64589411],
    #         [0.43758721, 0.891773, 0.96366276]]]) >,
    # < tf.Tensor: shape = (1, 3, 3), dtype = float64, numpy =
    # array([[[0.54488318, 0.4236548, 0.64589411],
    #         [0.43758721, 0.891773, 0.96366276],
    #         [0.38344152, 0.79172504, 0.52889492]]]) >,
    # < tf.Tensor: shape = (1, 3, 3), dtype = float64, numpy =
    # array([[[0.43758721, 0.891773, 0.96366276],
    #         [0.38344152, 0.79172504, 0.52889492],
    #         [0.56804456, 0.92559664, 0.07103606]]]) >]

