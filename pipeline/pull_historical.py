#!/usr/bin/env python
# due to IB limitation, this can only be ran locally where
# IB application is running
# this is to download historical data only and have the options of either place the flat file locally
# or upload to google storage
import argparse
import os
import datetime
import time
import threading
import logging
import pandas as pd
# from google.cloud import bigquery

from at_trading.gcp.config import GCP_PROJECT_ID
from at_trading.gcp.gcp_storage import storage_upload_file
from at_trading.ib.ib_api import gen_contract, ATIBApi

# *** ------- ****
# *** IB ONLY ****
# *** ------- ****
from at_trading.util.util_data_structure import to_parquet_table_from_df


def get_securities_dict_by_group(ib_app, securities_group):
    """

    :param ib_app:
    :param securities_group:
    :return:
    usage:
        >>> ib_app = ATIBApi()
        # app.connect_local(port=7497)
        >>> ib_app.connect_local()
        >>> api_thread = threading.Thread(target=ib_app.run, daemon=True)
        >>> api_thread.start()
        >>> ib_app.wait_till_connected()
    """
    result_sec_dict = {}
    if securities_group == 'vol':
        result_sec_dict['spx'] = gen_contract('SPX', 'IND', 'CBOE', 'USD')
        result_sec_dict['vix'] = gen_contract('VIX', 'IND', 'CBOE', 'USD')
        result_vix_futures = ib_app.req_contract_details_futures('VIX', exchange='CFE',
                                                                 summary_only=True)
        result_vix_futures_df = pd.DataFrame(result_vix_futures).sort_values('contract.lastTradeDateOrContractMonth')
        # take 5 front month contract
        result_vix_futures_df = result_vix_futures_df[result_vix_futures_df['marketName'] == 'VX'].head(5)
        for irow, data in result_vix_futures_df.iterrows():
            result_sec_dict[data['contract.localSymbol'].lower()] = gen_contract(
                symbol=data['contract.symbol'],
                sec_type=data['contract.secType'],
                exchange=data['contract.exchange'],
                currency=data['contract.currency'],
                other_param_dict={
                    'localSymbol': data['contract.localSymbol'],
                    'lastTradeDateOrContractMonth': data['contract.lastTradeDateOrContractMonth']
                }
            )
    elif securities_group == 'btc':
        bitcoin_futures = ib_app.req_contract_details_futures('BRR', exchange='CMECRYPTO',
                                                              summary_only=True)
        bitcoin_futures_df = pd.DataFrame(bitcoin_futures).sort_values('contract.lastTradeDateOrContractMonth')
        bitcoin_futures_df = bitcoin_futures_df[bitcoin_futures_df['marketName'] == 'BTC'].head(3)
        for irow, data in bitcoin_futures_df.iterrows():
            result_sec_dict[data['contract.localSymbol'].lower()] = gen_contract(
                symbol=data['contract.symbol'],
                sec_type=data['contract.secType'],
                exchange=data['contract.exchange'],
                currency=data['contract.currency'],
                other_param_dict={
                    'localSymbol': data['contract.localSymbol'],
                    'lastTradeDateOrContractMonth': data['contract.lastTradeDateOrContractMonth']
                }
            )

    return result_sec_dict


@DeprecationWarning
def create_bigquery_price_table_from_sample_file():
    client = bigquery.Client(project=GCP_PROJECT_ID)
    table_ref = client.dataset('market_data').table('ib_5sec_bar')

    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
    job_config.source_format = bigquery.SourceFormat.PARQUET
    uri = "gs://at-ml-bucket/prices/ib_5secs_trades_spx_20200728_20200729.parquet"
    load_job = client.load_table_from_uri(
        uri, table_ref, job_config=job_config
    )
    load_job.result()  # Waits for table load to complete.
    destination_table = client.get_table(table_ref)
    print("Loaded {} rows.".format(destination_table.num_rows))


def create_bigquery_price_table():
    logger = logging.getLogger(__name__)
    client = bigquery.Client(project=GCP_PROJECT_ID)
    dataset_ref = client.dataset('market_data')

    table_ref = dataset_ref.table("ib_5sec_bar")
    schema = [
        bigquery.SchemaField("date_time", "TIMESTAMP"),
        bigquery.SchemaField("open", "FLOAT"),
        bigquery.SchemaField("high", "FLOAT"),
        bigquery.SchemaField("low", "FLOAT"),
        bigquery.SchemaField("close", "FLOAT"),
        bigquery.SchemaField("volume", "INTEGER"),
        bigquery.SchemaField("barCount", "INTEGER"),
        bigquery.SchemaField("average", "FLOAT"),
        bigquery.SchemaField("ticker", "STRING"),
        bigquery.SchemaField("date", "DATE"),

    ]
    table = bigquery.Table(table_ref, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="date"
    )

    table = client.create_table(table)
    logger.info("Created table {}, partitioned on column {}".format(
        table.table_id, table.time_partitioning.field
    ))


def test_run(start_date, end_date, security_group):
    """

    :param security_group:
    :param start_date:
    :param end_date:
    usage:
    >>> start_date = '2020-09-21'
    >>> end_date = '2020-09-22'
    >>> security_group = 'btc'

    """
    logger = logging.getLogger(__name__)
    date_range = pd.date_range(start_date, end_date)
    app = ATIBApi()
    # app.connect_local(port=7497)
    app.connect_local()
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    app.wait_till_connected()

    for date_idx in range(len(date_range)):
        if date_idx != len(date_range) - 1:
            start_datetime = date_range[date_idx]
            end_datetime = date_range[date_idx + 1]
            logger.info('*******processing [{}] - [{}]*******'.format(start_datetime, end_datetime))
            securities_group = security_group
            data_source = 'ib'
            data_dir = '/Users/b3yang/workspace/temp/data'
            storage_path = 'prices'
            bar_size = '5 secs'
            data_type = 'TRADES'
            run_pull_historical_data(securities_group,
                                     data_source,
                                     start_datetime,
                                     end_datetime=end_datetime,
                                     bar_size=bar_size,
                                     data_type=data_type,
                                     data_dir=data_dir,
                                     upload_to_storage=True,
                                     storage_path=storage_path, ib_app=app)


def run_pull_historical_data(securities_group,
                             data_source,
                             start_datetime,
                             end_datetime=None,
                             bar_size='5 secs',
                             data_type='TRADES',
                             data_dir=None,
                             upload_to_storage=True,
                             storage_path='prices',
                             ib_app=None
                             ):
    """

    :param ib_app:
    :param securities_group:
    :param data_source:
    :param start_datetime:
    :param end_datetime:
    :param bar_size:
    :param data_type:
    :param data_dir:
    :param upload_to_storage:
    :param storage_path:
    usage:
        >>> securities_group = 'vol'
        >>> data_source = 'ib'
        >>> start_datetime = datetime.datetime(2020, 9, 21)
        >>> end_datetime = datetime.datetime(2020, 9, 22)
        >>> data_dir = '/Users/b3yang/workspace/temp/data'
        >>> upload_to_storage=True
        >>> storage_path = 'prices'
        >>> bar_size = '5 secs'
        >>> data_type = 'TRADES'
        >>> ib_app= None
    """
    if data_source.lower() == 'ib':
        if ib_app is None:
            app = ATIBApi()
            # app.connect_local(port=7497)
            app.connect_local()
            api_thread = threading.Thread(target=app.run, daemon=True)
            api_thread.start()
            app.wait_till_connected()
        else:
            app = ib_app
        # TODO: store to securities master database
        sec_dict = get_securities_dict_by_group(app, securities_group)
        if end_datetime is None:
            today_date = datetime.date.today()
            end_datetime = datetime.datetime(today_date.year, today_date.month, today_date.day)
        query_time = end_datetime.strftime('%Y%m%d %H:%M:%S')
        # find duration string
        duration_str = '{} D'.format((end_datetime - start_datetime).days)
        # TODO: test multiprocessing here
        for k, v in iter(sec_dict.items()):
            print('processing [{}]'.format(k))
            result = app.req_historical_data_blocking(v, query_time, duration_str,
                                                      bar_size, data_type, 1, 1, [])
            file_name = 'ib_{}_{}_{}_{}_{}.parquet'.format(bar_size.lower().replace(' ', ''),
                                                           data_type.lower(),
                                                           k,
                                                           start_datetime.strftime('%Y%m%d'),
                                                           end_datetime.strftime('%Y%m%d'))
            result_df = pd.DataFrame(result)
            result_df['ticker'] = k
            result_df = result_df.rename(columns={'date': 'date_time'})
            result_df['date_time'] = pd.to_datetime(result_df['date_time'])
            result_df['date'] = result_df['date_time'].dt.date

            to_parquet_table_from_df(result_df, data_dir, file_name)
            if upload_to_storage:
                storage_upload_file(bucket_name='at-ml-bucket',
                                    bucket_target_path=os.path.join(storage_path, file_name),
                                    full_path=os.path.join(data_dir, file_name),
                                    project_id=None)
            time.sleep(1)
        if ib_app is None:
            app.disconnect()
    else:
        raise Exception("only IB is supported at the moment")

#
# if __name__ == '__main__':
#     one_day = datetime.datetime.now() - datetime.timedelta(days=1)
#     one_day_date = datetime.datetime(one_day.year, one_day.month, one_day.day)
#
#     today_start = datetime.datetime.now()
#     today_start_date = datetime.datetime(today_start.year, today_start.month, today_start.day)
#
#     parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#
#     parser.add_argument(
#         '--securities-group',
#         default='vol',
#         help='Securities group, default to vol which contains VIX futures + VIX Index + SPX')
#     parser.add_argument(
#         '--data-source',
#         default='ib',
#         help='only supports IB at the moment')
#
#     parser.add_argument(
#         '--start-datetime',
#         default=one_day_date.strftime('%Y%m%d'),
#         help='start date time')
#
#     parser.add_argument(
#         '--end-datetime',
#         default=today_start_date.strftime('%Y%m%d'),
#         help='end date time')
#
#     parser.add_argument(
#         '--bar-size',
#         default='5 secs',
#         help='bar size')
#
#     parser.add_argument(
#         '--data-type',
#         default='TRADES',
#         help='data types')
#
#     parser.add_argument(
#         '--data-dir',
#         default='/Users/b3yang/workspace/temp/data/',
#         help='working directory for the parquet files being generated')
#
#     parser.add_argument(
#         '--upload-to-storage',
#         default=True,
#         help='upload to google storage')
#
#     parser.add_argument(
#         '--storage-path',
#         default='prices',
#         help='bucket folder under bucket at-ml-bucket')
#
#     args = parser.parse_args()
#
#     run_pull_historical_data(
#         securities_group=args.securities_group,
#         data_source=args.data_source,
#         start_datetime=datetime.datetime.strptime(args.start_datetime, '%Y%m%d'),
#         end_datetime=datetime.datetime.strptime(args.end_datetime, '%Y%m%d'),
#         bar_size=args.bar_size,
#         data_type=args.data_type,
#         data_dir=args.data_dir,
#         upload_to_storage=args.upload_to_storage,
#         storage_path=args.storage_path)
