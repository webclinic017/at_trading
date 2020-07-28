from typing import Dict, List


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
