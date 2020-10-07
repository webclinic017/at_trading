import cbpro
import datetime
import time
import pandas as pd
import os
import logging
from at_trading.gcp.gcp_secret_manager import access_secret_version
from at_trading.gcp.gcp_storage import storage_upload_file
from at_trading.util.util_data_structure import to_parquet_table_from_df


def get_cb_client(client_type,
                  key=None, secret=None,
                  passphrase=None,
                  api_url="https://api.pro.coinbase.com"):
    """

    :param client_type:
    :param key:
    :param secret:
    :param passphrase:
    :param api_url:
    :return:
    usage:
    >>> key='1b19ba867bf0e06f93f583b88825d485'
    >>> secret = 'TKCdKrstN+eX7KKZAhG08rPCh3Yo7kMcf9CUXEaE9TlQ/HAOzXely9UkdVv4PdVUtieW8eP97I9kXd0d4I2AGQ=='
    >>> passphrase = 'uihizupz01'
    >>> api_url = 'https://api.pro.coinbase.com'
    >>> client = get_cb_client('private', key, secret, passphrase)
    """
    if client_type == 'public':
        client = cbpro.PublicClient()
    else:
        client = cbpro.AuthenticatedClient(key, secret, passphrase,
                                           api_url=api_url)
    return client


def get_cb_clint_from_secret(api_url="https://api.pro.coinbase.com"):
    key = access_secret_version('at-ml-platform', 'coinbase-key', 'latest')
    secret = access_secret_version('at-ml-platform', 'coinbase-secret', 'latest')
    passphrase = access_secret_version('at-ml-platform', 'coinbase-pass', 'latest')

    client = get_cb_client('private',
                           key=key, secret=secret,
                           passphrase=passphrase,
                           api_url=api_url)
    return client


def get_product_historic_rates(start_date,
                               end_date,
                               data_dir,
                               client=None,
                               interval=60,
                               ticker='BTC-USD',
                               upload_to_storage=True,
                               storage_path='prices'):
    """

    :param client:
    :param storage_path:
    :param upload_to_storage:
    :param data_dir:
    :param ticker:
    :param start_date:
    :param end_date:
    :param interval:
    usage:
    >>> start_date = datetime.date(2020, 10, 5)
    >>> end_date = datetime.date(2020, 10, 6)
    >>> interval = 60
    >>> data_dir = '/Users/b3yang/workspace/temp/data'
    >>> ticker = 'BTC-USD'
    >>> upload_to_storage=True
    >>> storage_path = 'prices'
    """
    if client is None:
        client = get_cb_client('public')
    # assume start_date is date convert to datetime
    start_datetime = datetime.datetime(start_date.year, start_date.month, start_date.day)
    end_datetime = datetime.datetime(end_date.year, end_date.month, end_date.day)

    start_end_result_list = []
    while end_datetime > start_datetime:

        loop_end_datetime = start_datetime + datetime.timedelta(seconds=interval * 300)
        if loop_end_datetime > end_datetime:
            loop_end_datetime = end_datetime
        print('{}-{}'.format(start_datetime, loop_end_datetime))
        result_list = client.get_product_historic_rates(
            ticker,
            start_datetime,
            loop_end_datetime,
            interval
        )
        result_df = pd.DataFrame(result_list)
        result_df.columns = ['date_time', 'low', 'high', 'open', 'close', 'volume']
        result_df['date_time'] = pd.to_datetime(result_df['date_time'], unit='s')
        result_df = result_df.set_index('date_time').sort_index()
        start_end_result_list.append(result_df)
        start_datetime = loop_end_datetime
        time.sleep(0.5)

    combined_df = pd.concat(start_end_result_list).reset_index()
    combined_df['date'] = combined_df['date_time'].dt.date
    combined_df['ticker'] = ticker.replace('-', '').lower()
    file_name = 'cb_{}_{}_{}_{}_{}.parquet'.format('{}secs'.format(interval),
                                                   'trades',
                                                   ticker.replace('-', '').lower(),
                                                   start_date.strftime('%Y%m%d'),
                                                   end_date.strftime('%Y%m%d'))
    to_parquet_table_from_df(combined_df, data_dir, file_name)
    if upload_to_storage:
        storage_upload_file(bucket_name='at-ml-bucket',
                            bucket_target_path=os.path.join(storage_path, file_name),
                            full_path=os.path.join(data_dir, file_name),
                            project_id=None)


def test_run(start_date, end_date):
    """

    :param start_date:
    :param end_date:
    usage:
    >>> start_date = '2020-09-16'
    >>> end_date = '2020-10-06'

    """
    logger = logging.getLogger(__name__)
    date_range = pd.date_range(start_date, end_date)
    client = get_cb_clint_from_secret()

    for date_idx in range(len(date_range)):
        if date_idx != len(date_range) - 1:
            start_datetime = date_range[date_idx]
            end_datetime = date_range[date_idx + 1]
            logger.info('*******processing [{}] - [{}]*******'.format(start_datetime, end_datetime))
            data_dir = '/Users/b3yang/workspace/temp/data'
            storage_path = 'prices'
            get_product_historic_rates(start_datetime,
                                       end_datetime,
                                       data_dir,
                                       client=client,
                                       interval=60,
                                       ticker='BTC-USD',
                                       upload_to_storage=True,
                                       storage_path=storage_path)
