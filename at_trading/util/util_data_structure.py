from typing import Dict, List
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
import uuid
import logging


def to_parquet_table_from_df(input_df: pd.DataFrame, file_path: str = None,
                             file_name: str = None, arg_dict: dict = None) -> str:
    """

    :param arg_dict:
    :param input_df:
    :param file_path:
    :param file_name:
    usage:
        >>> input_df = pd.DataFrame({'a':[1,2, 3],'b':[2, 3, 4]})
        >>> file_path = None
        >>> file_name = None

        >>> file_path = '/Users/b3yang/workspace/at_data/'
        >>> file_name = 'blah.parquet'
    """
    logger = logging.getLogger(__name__)
    f_path = '.' if file_path is None else file_path
    f_name = '{}.parquet'.format(str(uuid.uuid1())) if file_name is None else file_name
    args = {} if arg_dict is None else arg_dict

    final_path = os.path.join(f_path, f_name)
    try:
        table = pa.Table.from_pandas(input_df, preserve_index=False)
        pq.write_table(table, final_path, **args)

    except Exception as e:
        logger.error(e)
    logger.info('[{}] written'.format(final_path))
    return final_path


def flatten_dictionary(input_dict: Dict, filter_list: List):
    """

    :param input_dict:
    :param filter_list:
    :return:
    usage:
    >>> input_dict = {
    'a': 123,
    'b': {'something':'blah'},
    'c': 'great'}
    >>> filter_list = ['c', 'b.something']
    >>> flatten_dictionary(input_dict, filter_list)
    Out[50]: {'c': 'great', 'b.something': 'blah'}

    nested dictionary
    >>> input_dict = {
    'a': 123,
    'b': {'something':{
    'nested': 'finally here',
    'nested2': 'finally here2'
    }},
    'c': 'great'}
    >>> filter_list = ['c', 'b.something.nested2']
    >>> flatten_dictionary(input_dict, filter_list)

    """
    result_dict = {}
    for f in filter_list:
        if '.' in f:
            split_list = f.split('.')
            # handle multiple levels of nesting
            if len(split_list) > 2:
                rejoined = '.'.join(split_list[1:])
                recursive_result = flatten_dictionary(input_dict[split_list[0]], [rejoined])
                result_dict[f] = recursive_result[rejoined]
            elif len(split_list) == 2:
                result_dict[f] = input_dict[split_list[0]][split_list[1]]
            else:
                result_dict[f] = input_dict[f]
        else:
            result_dict[f] = input_dict[f]
    return result_dict
