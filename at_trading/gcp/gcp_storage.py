# storage
import logging
from google.cloud import storage
from at_trading.gcp.config import GCP_PROJECT_ID


def storage_gen_uri(bucket_name: str, bucket_target_path: str, file_filter: str = None) -> str:
    """

    :param bucket_name:
    :param bucket_target_path:
    :param file_filter:
    :return:
    usage:
    >>> bucket_name = 'qfs_at'
    >>> bucket_target_path = 'united_states'
    >>> file_filter = '*.csv'
    """
    result_path = 'gs://{}/{}/{}'.format(bucket_name, bucket_target_path, file_filter)
    return result_path


def storage_create_bucket(bucket_name: str, project_id: str = None):
    """

    :param bucket_name:
    :param project_id:
    """
    logger = logging.getLogger(__name__)
    p_id = GCP_PROJECT_ID if project_id is None else project_id
    try:
        storage_client = storage.Client(project=p_id)
        storage_client.create_bucket(bucket_name)
        logger.info('bucket {} created'.format(bucket_name))
    except Exception as e:
        logger.error(e)


def storage_upload_file(bucket_name: str, bucket_target_path: str, full_path: str, project_id: str = None):
    """

    :param bucket_name:
    :param bucket_target_path:
    :param full_path:
    :param project_id:
    usage:
    >>> bucket_target_path = 'metrics/definition.parquet'
    >>> full_path = ''
    """
    logger = logging.getLogger(__name__)
    p_id = GCP_PROJECT_ID if project_id is None else project_id
    try:
        storage_client = storage.Client(project=p_id)
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(bucket_target_path)
        blob.upload_from_filename(full_path)
        logger.info('[{}/{}] uploaded'.format(bucket_name, bucket_target_path))
    except Exception as e:
        logger.error(e)
