import tensorflow as tf
import functools

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

# Referenced implementation of WGAN-GP (https://github.com/igul222/improved_wgan_training.git)
def subpixelconv(filters, kernel_size = (1, 1), kernel_initializer = 'he'):
    def f(x):
        x = tf.layers.Conv2D(4*filters, kernel_size = kernel_size,
                             padding = 'same', kernel_initializer = kernel_initializer)(x)
        x = tf.depth_to_space(x, 2)
        return x
    return f

def meanpoolconv(filters, kernel_size = (1, 1), kernel_initializer = 'he'):
    def f(x):
        x = tf.add_n([x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]])/4.
        x = tf.layers.Conv2D(filters, kernel_size = kernel_size,
                             padding = 'same', kernel_initializer = kernel_initializer)(x)
        return x
    return f

def convmeanpool(filters, kernel_size = (1, 1), kernel_initializer = 'he'):
    def f(x):
        x = tf.layers.Conv2D(filters, kernel_size = kernel_size,
                             padding = 'same', kernel_initializer = kernel_initializer)(x)
        x = tf.add_n([x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]])/4.
        return x
    return f

def upsamplingconv(filters, kernel_size = (1, 1), kernel_initializer = 'he'):
    def f(x):
        x = tf.concat([x, x, x, x], axis = 3)
        x = tf.depth_to_space(x, 2)
        x = tf.layers.Conv2D(filters, kernel_size = kernel_size,
                             padding = 'same', kernel_initializer = kernel_initializer)(x)
        return x
    return f


class PlainBlock(object):
    def __init__(self, filters, kernel_size, batch_norm = True,
                 resample = None, name = 'plain'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.resample = resample
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _, n, h, w = tf.unstack(tf.shape(x))

            if self.resample == 'down':
                shortcut_layer = meanpoolconv
                conv_1 = functools.partial(tf.layers.Conv2D, filters = n)
                conv_2 = functools.partial(convmeanpool, filters = self.filters)
            elif self.resample == 'up':
                shortcut_layer = upsamplingconv
                conv_1 = functools.partial(upsamplingconv, filters = self.filters)
                conv_2 = functools.partial(tf.layers.Conv2D, filters = self.filters)
            elif self.resample == None:
                shortcut_layer = functools.partial(tf.layers.Conv2D, padding = 'same')
                conv_1 = functools.partial(tf.layers.Conv2D, filters = n)
                conv_2 = functools.partial(tf.layers.Conv2D, filters = self.filters)
            else:
                raise Exception('invalid resample value')

            if self.filters == n and resample = None:
                shortcut = x
            else:
                shortcut = shortcut_layer(self.filters, kernel_size = (1, 1),
                                          kernel_initializer = None)(x)

            if self.batch_norm: x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
            x = conv_1(kernel_size = self.kernel_size, padding = 'same'
                       kernel_initializer = 'he')(x)
            if self.batch_norm: x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
            x = conv_2(kernel_size = self.kernel_size, padding = 'same'
                       kernel_initializer = 'he')(x)

            return shortcut + x


class BottleneckBlock(object):
    def __init__(self, filters, kernel_size, batch_norm = True,
                 resample = None, name = 'bottleneck'):
        self.filters = filters
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.resample = resample
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            _, n, h, w = tf.unstack(tf.shape(x))

            '''resample: None, 'up', 'down' '''
            if self.resample == 'down':
                shortcut_layer = functools.partial(tf.layers.Conv2D, strides = (2, 2),
                                                   padding = 'same')
                conv_1 = functools.partial(tf.layers.Conv2D, filters = n/2)
                conv_b = functools.partial(tf.layers.Conv2D, filters = self.filters/2,
                                           strides = (2, 2))
                conv_2 = functools.partial(tf.layers.Conv2D, filters = self.filters)
            elif self.resample == 'up':
                shortcut_layer = subpixelconv
                conv_1 = functools.partial(tf.layers.Conv2D, filters = n/2)
                conv_b = functools.partial(tf.layers.Conv2DTranspose, filters = self.filters/2,
                                           strides = (2, 2))
                conv_2 = functools.partial(tf.layers.Conv2D, filters = self.filters)
            elif self.resample == None:
                shortcut_layer = functools.partial(tf.layers.Conv2D, padding = 'same')
                conv_1 = functools.partial(tf.layers.Conv2D, filters = n/2)
                conv_b = functools.partial(tf.layers.Conv2D, filters = self.filters/2)
                conv_2 = functools.partial(tf.layers.Conv2D, filters = self.filters)

            else:
                raise Exception('invalid resample value')

            if filters == n and resample == None:
                shortcut = x
            else:
                shortcut = shortcut_layer(self.filters, kernel_size = (1, 1),
                                          kernel_initializer = None)(x)

            x = tf.nn.relu(x)
            x = conv_1(kernel_size = (1, 1), padding = 'same',
                       kernel_initializer = 'he')(x)
            x = tf.nn.relu(x)
            x = conv_b(kernel_size = self.kernel_size, padding = 'same',
                       kernel_initializer = 'he')(x)
            x = tf.nn.relu(x)
            x = conv_2(kernel_size = (1, 1), padding = 'same',
                       kernel_initializer = 'he')(x)
            if self.batch_norm:
                x = tf.layers.BatchNormalization()(x)

            return shortcut + (0.3*x)
            
            

