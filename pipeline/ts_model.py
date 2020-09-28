import tensorflow as tf


def get_model_lstm(n_features,
                   n_out_steps: int = 1,
                   lstm_units: int = 32,
                   n_output: int = None,
                   return_sequences: bool = False):
    """

    :param n_output:
    :param n_features:
    :param n_out_steps:
    :param lstm_units:
    :param return_sequences:
    :return:
    """
    if n_output is None:
        _n_output = n_features
    else:
        _n_output = n_output

    layer_list = [tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences),
                  tf.keras.layers.Dense(n_out_steps * n_features,
                                        kernel_initializer=tf.initializers.zeros)]
    if n_features != _n_output:
        layer_list.append(tf.keras.layers.Dense(n_out_steps * _n_output))

    layer_list.append(tf.keras.layers.Reshape([n_out_steps, _n_output]))

    model = tf.keras.Sequential(layer_list)
    return model
