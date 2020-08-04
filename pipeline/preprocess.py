import argparse
from apache_beam.options.pipeline_options import PipelineOptions
import re
from google.cloud import bigquery
import logging
import pandas as pd
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


def run_preprocess_pipeline_batch(bucket_full_path, pipeline_options, save_main_session=True):
    """

    :param bucket_full_path:
    :param pipeline_options:
    :param save_main_session:
    :return:
    """
    pipeline_options = PipelineOptions(pipeline_options)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    with beam.Pipeline(options=pipeline_options) as p:
        dataframes = p \
                     | 'Read' >> ReadFromParquetBatched(bucket_full_path)\
                     | 'Convert to pandas' >> beam.Map(lambda table: table.to_pandas())
        result = p.run()
        result.wait_until_finish()
        print(result)

        #
        # # Count the occurrences of each word.
        # counts = (
        #     lines
        #     | 'Split' >> (
        #         beam.FlatMap(lambda x: re.findall(r'[A-Za-z\']+', x)).
        #             with_output_types(unicode))
        #     | 'PairWithOne' >> beam.Map(lambda x: (x, 1))
        #     | 'GroupAndSum' >> beam.CombinePerKey(sum))
        #
        # # Format the counts into a PCollection of strings.
        # def format_result(word_count):
        #     (word, count) = word_count
        #     return '%s: %s' % (word, count)
        #
        # output = counts | 'Format' >> beam.Map(format_result)
        #
        # # Write the output using a "Write" transform that has side effects.
        # # pylint: disable=expression-not-assigned
        # output | WriteToText(known_args.output)


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
        # CHANGE 1/6: The Google Cloud Storage path is required
        # for outputting the results.
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
    run_preprocess_pipeline_batch(known_args.input, pipeline_args, True)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
