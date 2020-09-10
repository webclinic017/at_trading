from typing import Dict

import pandas as pd

from pipeline.ts_sample_data import make_sample_data_ts


def split_data_frame(input_df: pd.DataFrame,
                     split_dict: Dict = None) -> Dict[str, pd.DataFrame]:
    """

    :param input_df:
    :param split_dict:
    usage:
    >>> input_df = make_sample_data_ts(read_from_bucket=False)
    >>> split_data_frame(input_df)
    """
    n_items = len(input_df)
    result_dict = {}
    if split_dict is None:
        s_dict = {
            'train': 0.6,
            'val': 0.2,
            'test': 0.2
        }
    else:
        s_dict = split_dict

    accum_so_far = 0.0
    for idx, (key, value) in enumerate(s_dict.items()):
        if idx == 0:
            split_df = input_df[0:int(n_items * value)]
        elif idx == len(s_dict) - 1:
            split_df = input_df[int(n_items * accum_so_far):]
        else:
            split_df = input_df[int(n_items * accum_so_far):int(n_items * (accum_so_far + value))]
        accum_so_far += value
        result_dict[key] = split_df

    return result_dict
