import tensorflow as tf

from pipeline.ts_feature import split_data_frame
from pipeline.ts_sample_data import make_sample_data_ts
from pipeline.ts_window import TSWindowGenerator


def compile_and_fit(model,
                    window,
                    patience=2,
                    max_epochs=20):
    """

    :param model:
    :param window:
    :param patience:
    :param max_epochs:
    :return:
    usage:
    >>> model = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
    ])
    >>> input_width = 100
    >>> shift = 5
    >>> label_width = 2
    >>> all_df = make_sample_data_ts(read_from_bucket=True)
    >>> result_dict = split_data_frame(all_df)
    >>> train_df = result_dict['train']
    >>> val_df = result_dict['val']
    >>> test_df = result_dict['test']
    >>> window = TSWindowGenerator(
            input_width,
             label_width,
             shift,
             train_df,
             val_df,
             test_df,
             None)
    >>> patience = 2
    >>> max_epochs = 20

    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train[0],
                        epochs=max_epochs,
                        validation_data=window.val[0],
                        callbacks=[early_stopping])
    return history
