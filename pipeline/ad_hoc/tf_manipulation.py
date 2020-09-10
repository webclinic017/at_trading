import tensorflow as tf

# remove column or slice
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
slice1 = tf.slice(a, [0, 0], [2, 1])
slice2 = tf.slice(a, [0, 2], [2, 1])
a_new = tf.concat([slice1, slice2], 1)

# slicing
# sample here shows 3, 2, 3
# <tf.Tensor: shape=(3, 2, 3), dtype=int32, numpy=
# array([[[1, 1, 1],
#         [2, 2, 2]],
#        [[3, 3, 3],
#         [4, 4, 4]],
#        [[5, 5, 5],
#         [6, 6, 6]]], dtype=int32)>

t = tf.constant([[[1, 1, 1], [2, 2, 2]],
                 [[3, 3, 3], [4, 4, 4]],
                 [[5, 5, 5], [6, 6, 6]]])


tf.slice(t, [0, 0, 0], [1, 1, 3])  # [[[3, 3, 3]]]
tf.slice(t, [1, 0, 0], [1, 2, 3])  # [[[3, 3, 3],
#   [4, 4, 4]]]
tf.slice(t, [1, 0, 0], [2, 1, 3])  # [[[3, 3, 3]],
#  [[5, 5, 5]]]
