import tensorflow as tf
import tensorflow_probability as tfp

# @tf.function
def clip(x_in):
    median = tfp.stats.percentile(x_in, 50.0)
    total_var = tf.reduce_mean(tf.math.abs(x_in-median))
    clip_min = median - 5*total_var
    clip_max = median + 5*total_var

    x_out = tf.clip_by_value(x_in, clip_min, clip_max)
    return tf.reshape(x_out, x_in.shape)
