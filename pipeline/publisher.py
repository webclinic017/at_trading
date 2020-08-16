#!/usr/bin/env python

# publisher that publish events to gcp pubsub queue
import pandas as pd
import tensorflow as tf
import json
from at_trading.gcp.config import GCP_PROJECT_ID
import logging
import time

from at_trading.gcp.gcp_pubsub import gcp_pubsub_create_topic, gcp_pubsub_publish


def publish_message_from_storage(bucket_path,
                                 topic_id,
                                 create_topic=False,
                                 project_id=None):
    """

    currently a fake publisher that reads from flat parquet file and publishes the msg directly
    to ib_stream pubsub topic

    TODO:
        make it compatible scheduler to read from storage directly and publish the data
        therefore whenever there's no batch file dropped, we can stream the data to pubsub
        and hence automatilcally trigger the recalibration of the model

    :param project_id:
    :param topic_id:
    :param create_topic:
    :param bucket_path:
    :return:
    usage:
    >>> bucket_path = 'gs://at-ml-bucket/prices/*.parquet'
    >>> topic_id = 'ib_stream'
    >>> create_topic = False
    >>> project_id = None


    """
    logger = logging.getLogger(__name__)
    if project_id is None:
        project_id = GCP_PROJECT_ID

    if create_topic:
        gcp_pubsub_create_topic(topic_id, project_id=project_id)
    file_name_list = tf.io.gfile.glob(bucket_path)
    for f in file_name_list:
        # drop the date column for now since it's really not needed
        df = pd.read_parquet(f).drop('date', axis=1)

        df['date_time'] = df['date_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        # publish a message for each row
        for idx, row in df.iterrows():
            json_msg = json.dumps(row.to_dict())
            gcp_pubsub_publish(topic_id, json_msg, project_id=project_id)
            # wait a sec
            time.sleep(1)
            logger.info('streaming [{}]'.format(json_msg))
