import argparse
import pyarrow as pa
from os import path
from apache_beam.options.pipeline_options import PipelineOptions
import re
from google.cloud import bigquery
import logging
import pandas as pd
from pyarrow.parquet import write_table

from at_trading.gcp.config import GCP_PROJECT_ID
from past.builtins import unicode

import apache_beam as beam
from apache_beam.io import ReadFromParquetBatched
from apache_beam.io import WriteToText
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions


def populate_table_from_parquet(bucket_name='at-ml-bucket', storage_path='prices'):
    """
    TODO:
        we might not need populate bigquery tables, so ignore for now
        later when we are combining the historical/streaming data
        we can come back to this

    :param bucket_name:
    :param storage_path:
    usage:
    >>> bucket_name='at-ml-bucket'
    >>> storage_path='prices'
    """
    logger = logging.getLogger(__name__)
    client = bigquery.Client(project=GCP_PROJECT_ID)
    table_ref = client.dataset('market_data').table('ib_5sec_bar')
    job_config = bigquery.LoadJobConfig()
    job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    job_config.source_format = bigquery.SourceFormat.PARQUET

    uri = "gs://{}/{}/ib_5secs_trades_*.parquet".format(bucket_name, storage_path)
    load_job = client.load_table_from_uri(
        uri, table_ref, job_config=job_config
    )
    logger.info("Starting job {}".format(load_job.job_id))
    load_job.result()
    logger.info('job finished')

    destination_table = client.get_table(table_ref)
    logger.info("Loaded {} rows.".format(destination_table.num_rows))


class ComputeReturn(beam.DoFn):
    def to_runner_api_parameter(self, unused_context):
        pass

    def process(self, element):
        assert isinstance(element, pd.DataFrame), \
            'element is of type [{}], only dataframe is supported'.format(
                type(element))

        sorted_element = element.sort_values('date_time')
        sorted_element['period_return'] = sorted_element['close'].pct_change()
        return_element = sorted_element[['date_time', 'ticker', 'period_return']].dropna()
        return_element = return_element.set_index('date_time')
        yield return_element


def combine_all_dataframe(dataframe_list):
    feature_df = None
    if dataframe_list:
        feature_df = pd.concat(dataframe_list)
    return feature_df


class WriteToBucket(beam.DoFn):

    def to_runner_api_parameter(self, unused_context):
        pass

    def __init__(self, bucket_path, *unused_args, **unused_kwargs):
        super().__init__(*unused_args, **unused_kwargs)
        self.output_path = bucket_path

    def process(self, element):
        full_path = path.join(self.output_path, 'test.parquet')
        element.pivot(columns='ticker', values='period_return').dropna().to_parquet(full_path)


def run_preprocess_pipeline_batch(bucket_full_path,
                                  output_path,
                                  pipeline_options,
                                  start_date,
                                  end_date,
                                  save_main_session=True):
    """

    :param output_path:
    :param end_date:
    :param start_date:
    :param bucket_full_path:
    :param pipeline_options:
    :param save_main_session:
    :return:
    """
    pipeline_options = PipelineOptions(pipeline_options)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session
    pipeline = beam.Pipeline(options=pipeline_options)

    dataframes = (pipeline
                  | 'Read' >> ReadFromParquetBatched(bucket_full_path)
                  | 'Convert to pandas' >> beam.Map(lambda table: table.to_pandas())
                  | 'calculate return' >> beam.ParDo(ComputeReturn())
                  | 'combine into feature dataframe' >> beam.CombineGlobally(combine_all_dataframe)
                  | 'write the result to bucket' >> beam.ParDo(WriteToBucket(output_path))
                  )

    result = pipeline.run()
    result.wait_until_finish()


def run(argv=None, save_main_session=True):
    """Main entry point; defines and runs the wordcount pipeline."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        default='gs://at-ml-bucket/prices/*.parquet',
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        default='gs://at-ml-bucket/beam/output',
        help='Output file to write results to.')
    known_args, pipeline_args = parser.parse_known_args(argv)

    pipeline_args.extend([
        # DataflowRunner or DirectRunner
        '--runner=DirectRunner',
        '--project={}'.format(GCP_PROJECT_ID),
        '--region={}'.format('us-central1'),
        '--staging_location=gs://at-ml-bucket/beam/staging',
        '--temp_location=gs://at-ml-bucket/beam/temp',
        '--job_name=your-wordcount-job',
    ])
    run_preprocess_pipeline_batch(known_args.input, known_args.output, pipeline_args, None, None, True)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
