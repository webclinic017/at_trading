import pandas as pd
from google.cloud import storage
import logging
import seaborn as sns
import datetime
from pipeline.ts_feature import split_data_frame
import numpy as np

from pipeline.ts_model import get_model_lstm
from pipeline.ts_window import TSWindowGenerator, compile_and_fit


def load_data(return_raw_and_pct_change=True):
    """
    # 1. load data into a dataframe and break it down to tensorflow dataset
    # TODO: read into dataset directly instead of using pandas since it's easier to process large data
    # beam also provide an interface that can be used as part of the pipeline
    """
    logger = logging.getLogger(__name__)
    client = storage.Client()
    df_list = []
    for blob in client.list_blobs('at-ml-bucket', prefix='prices'):
        f = 'gs://at-ml-bucket/{}'.format(blob.name)
        logger.info('loading [{}]'.format(blob.name))
        df_loaded = pd.read_parquet(f)[['date_time', 'close', 'ticker']].set_index('date_time')
        df_list.append(df_loaded)
    df_combined = pd.concat(df_list)
    df_combined = df_combined.pivot(columns='ticker', values='close')
    df_combined = df_combined[~df_combined['spx'].isnull()].fillna(method='ffill')

    df_pct_change = df_combined.pct_change().dropna()
    if return_raw_and_pct_change:
        df_pct_change = df_pct_change.rename(columns={k: '{}_pct'.format(k) for k in df_pct_change.columns})
        return pd.concat([df_combined, df_pct_change], axis=1).dropna()
    else:
        return df_pct_change


def feature_engineering(input_df):
    """
    # 2. feature engineering
    - visualization
    - split to train/valuation/test set
    - normalization
    - convert timestamp into seconds and apply sin + cos such that it wraps around and treat it as feature

    usage:
    >>> input_df = load_data()
    """

    # convert timestamps
    sec_in_day = 24 * 60 * 60
    time_stamp = input_df.index.map(datetime.datetime.timestamp)
    input_df['timestamp_sin'] = np.sin(time_stamp * (2 * np.pi / sec_in_day))
    input_df['timestamp_cos'] = np.cos(time_stamp * (2 * np.pi / sec_in_day))

    # clearly cannot use percentage change or level as is, need to standardize, but let's split data first
    result_dict = split_data_frame(input_df)
    train_df = result_dict['train']
    val_df = result_dict['val']
    test_df = result_dict['test']

    train_mean = train_df.mean()
    train_std = train_df.std()
    norm_input_df = input_df.sub(train_mean).div(train_std)
    ax = sns.violinplot(data=norm_input_df)
    _ = ax.set_xticklabels(input_df.columns, rotation=45)

    norm_train = train_df.sub(train_mean).div(train_std)
    norm_val = val_df.sub(train_mean).div(train_std)
    norm_test = test_df.sub(train_mean).div(train_std)
    return norm_train, norm_val, norm_test


def create_window(train_df,
                  val_df,
                  test_df,
                  input_width=1200,
                  shift=1,
                  label_width=1,
                  label_columns=None):
    window = TSWindowGenerator(
        input_width,
        label_width,
        shift,
        train_df,
        val_df,
        test_df,
        label_columns=label_columns)
    return window


def create_model(model_type, n_features, n_out_steps):
    model = None
    if model_type == 'lstm':
        model = get_model_lstm(n_features, n_out_steps)

    return model


def run_driver():
    input_df = load_data(return_raw_and_pct_change=True)
    train_df, val_df, test_df = feature_engineering(input_df)
    window = create_window(train_df, val_df, test_df, label_columns=['vxq0_pct'])
    model = create_model('lstm', len(input_df.columns), 1)
    compile_and_fit(model, window, max_epochs=5)
    window.plot_train_data(model=model, column_filters=['vxq0_pct'])

