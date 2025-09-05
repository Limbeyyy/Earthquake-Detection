import tensorflow as tf

def wasserstein_1D(a, b):
    return tf.reduce_mean(tf.abs(tf.sort(a) - tf.sort(b)), axis=-1)

def euclidean(a, b):
    return tf.sqrt(tf.reduce_sum(tf.square(a - b), axis=-1))
    
