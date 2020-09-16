import tensorflow as tf


def get_model_lstm(n_features,
                   n_out_steps: int = 1,
                   lstm_units: int = 32,
                   return_sequences: bool = False):
    """

    :param n_features:
    :param n_out_steps:
    :param lstm_units:
    :param return_sequences:
    :return:
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(lstm_units, return_sequences=return_sequences),
        tf.keras.layers.Dense(n_out_steps * n_features,
                              kernel_initializer=tf.initializers.zeros),
        tf.keras.layers.Reshape([n_out_steps, n_features])
    ])
    return model
