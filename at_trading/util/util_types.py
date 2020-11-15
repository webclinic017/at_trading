primitive = (int, str, bool, float)


def is_primitive(input_obj):
    return isinstance(input_obj, primitive)
