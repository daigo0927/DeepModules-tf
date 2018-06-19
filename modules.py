import tensorflow as tf

def conv_block(filters, kernel_size = (3, 3), strides = (1, 1), batch_norm = False):
    def f(x):
        x = tf.layers.Conv2D(filters, kernel_size,
                             strides, 'same')(x)
        if batch_norm:
            x = tf.layers.BatchNormalization()(x)
        x = tf.nn.leaky_relu(x, 0.2)
        return x
    return f
    
def bn_relu_conv(filters, kernel_size = (3, 3), strides = (1, 1)):
    def f(x):
        x = tf.layers.BatchNormalization()(x)
        x = tf.nn.relu(x)
        x = tf.layers.Conv2D(filters, kernel_size,
                             strides, 'same')(x)
        return x
    return f

def shortcut(x, fx):
    pass
