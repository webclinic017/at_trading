import os
from google.cloud import pubsub_v1
import logging

from at_trading.gcp.config import GCP_PROJECT_ID


def gcp_pubsub_create_topic(topic_id, project_id=None):
    """

    :param topic_id:
    :param project_id:
    usage:
    this would work after IAM role is updated (i.e. add service account to Pub/Sub)
    >>> gcp_pubsub_create_topic('ib_prices')
    """
    logger = logging.getLogger(__name__)
    if project_id is not None:
        p_id = project_id
    else:
        p_id = GCP_PROJECT_ID
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(p_id, topic_id)
    topic = publisher.create_topic(topic_path)
    logger.info('topic created [{}]'.format(topic))


def gcp_pubsub_publish(topic_id, message, message_param_dict=None, project_id=None):
    """

    :param topic_id:
    :param message:
    :param message_param_dict:
    :param project_id:
    usage:
    >>> topic_id = 'ib_prices'
    >>> message = 'TEST'
    >>> message_param_dict = None
    >>> project_id = None
    """
    if project_id is not None:
        p_id = project_id
    else:
        p_id = GCP_PROJECT_ID

    if message_param_dict is None:
        param_dict = {}
    else:
        param_dict = message_param_dict

    publisher = pubsub_v1.PublisherClient()
    topic_name = 'projects/{project_id}/topics/{topic}'.format(
        project_id=p_id,
        topic=topic_id
    )
    publisher.publish(topic_name,
                      message.encode('utf-8'), **param_dict)
