#!/usr/bin/env python

# *********
# pipeline that will do the following
# 1. reads pubsub msg from a topic
# 2. create a window size and aggregate all messages within the window size
# 3. compute the h/l/o/c and any other information needed
# 4. publish to storage AND publish to another pubsub topic
# (where another dataflow service will pick it up and do predictions on it)
# *********
import pandas as pd
import argparse
import json
import logging
from at_trading.gcp.config import GCP_PROJECT_ID
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
import apache_beam.transforms.window as window


class ExposeMsgTimestamp(beam.DoFn):
    def to_runner_api_parameter(self, unused_context):
        pass

    def process(self, element):
        item = element.decode("utf-8")
        item_json = json.loads(item)
        yield {
            "message_body": item,
            "publish_time": item_json['date_time']
        }


class AggregateData(beam.DoFn):
    def to_runner_api_parameter(self, unused_context):
        pass

    def process(self, element):
        msg_body_list = [json.loads(x['message_body']) for x in element]
        agg_df = pd.DataFrame(msg_body_list)
        agg_df['date_time'] = pd.to_datetime(agg_df['date_time'])
        agg_df = agg_df.sort_values('date_time')
        result_dict = {
            'low': agg_df['close'].min(),
            'high': agg_df['close'].min(),
            'close': agg_df['close'].iloc[-1],
            'open': agg_df['close'].iloc[0],
            'date_time': agg_df['date_time'].max().strftime('%Y-%m-%d %H:%M:%S')
        }

        print(result_dict)
        yield result_dict


class GroupWindowsIntoBatches(beam.PTransform):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = int(window_size)

    def expand(self, input_or_inputs):
        return (
            input_or_inputs
            | "Window into fixed timespan" >> beam.WindowInto(window.FixedWindows(self.window_size))
            # add timestamps, in this case, read it directly from the message
            | "Add timestamps to messages" >> beam.ParDo(ExposeMsgTimestamp())
            # dummy key is used for aggregation purpose, i.e. every self.window_size we can assign a unique key
            | "Add Dummy Key" >> beam.Map(lambda elem: (None, elem))
            | "Groupby" >> beam.GroupByKey()
            | "Abandon Dummy Key" >> beam.MapTuple(lambda _, val: val)
            | "Aggregate HLOC" >> beam.ParDo(AggregateData())
        )


class WriteToBucket(beam.DoFn):
    def to_runner_api_parameter(self, unused_context):
        pass

    def __init__(self, bucket_path, *unused_args, **unused_kwargs):
        super().__init__(*unused_args, **unused_kwargs)
        self.output_path = bucket_path

    def process(self, element, window_param=beam.DoFn.WindowParam):
        ts_format = "%H:%M:%S"
        window_start = window_param.start.to_utc_datetime().strftime(ts_format)
        window_end = window_param.end.to_utc_datetime().strftime(ts_format)
        filename = "-".join([self.output_path, window_start, window_end]) + '.json'

        with beam.io.gcp.gcsio.GcsIO().open(filename=filename, mode="w") as f:
            f.write("{}\n".format(json.dumps(element)).encode("utf-8"))


def run_stream(topic_id, output_path, pipeline_options, window_size=120, save_main_session=True):
    pipeline_options = PipelineOptions(pipeline_options, streaming=True)
    pipeline_options.view_as(SetupOptions).save_main_session = save_main_session

    with beam.Pipeline(options=pipeline_options) as p:
        _ = p \
            | 'Read' >> beam.io.ReadFromPubSub(topic=topic_id) \
            | "Window into" >> GroupWindowsIntoBatches(window_size) \
            | "Write to gcs" >> beam.ParDo(WriteToBucket(output_path))


def run(argv=None, save_main_session=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--topic',
        dest='topic',
        default='projects/{}/topics/{}'.format(GCP_PROJECT_ID, 'ib_stream'))
    parser.add_argument(
        '--output',
        dest='output',
        default='gs://at-ml-bucket/ib/streams-',
        help='Output file to write results to.')
    parser.add_argument(
        '--window_size',
        dest='window_size',
        default=10)

    known_args, pipeline_args = parser.parse_known_args(argv)
    pipeline_args.extend([
        # DataflowRunner or DirectRunner
        '--runner=DataflowRunner',
        '--project={}'.format(GCP_PROJECT_ID),
        '--region={}'.format('us-central1'),
        '--staging_location=gs://at-ml-bucket/beam/staging',
        '--temp_location=gs://at-ml-bucket/beam/temp',
        '--job_name=ib-streaming-job'
    ])
    run_stream(known_args.topic, known_args.output, pipeline_args,
               known_args.window_size,
               save_main_session=save_main_session)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    run()
