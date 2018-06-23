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
def he_normal(shape): # kernel_shape (ksize, ksize, channel_in, channel_out)
    fan_in = tf.reduce_prod(shape[::-1]) # ksize*ksize*channel_in
    fan_in = tf.cast(fan_in, dtype = tf.float32)
    # 0.879... = scipy.stats.truncnorm.std(-2, 2, 0., 1.), stated in keras documents.
    stddev = tf.sqrt(2./fan_in) / .87962566103423978
    return tf.initializers.truncated_normal(0.0, stddev)

def Conv2D(in_channels, out_channels, kernel_size, strides = (1, 1),
           padding = 'same', kernel_initializer = None):
    def f(x):
        if kernel_initializer is 'he_normal':
            kernel_shape = list(kernel_size) + [in_channels, out_channels]
            initializer = he_normal(kernel_shape)
        elif kernel_initializer is None:
            initializer = None
        else:
            raise Exception('invalid argument for kernel initializer')
        
        x = tf.layers.Conv2D(out_channels, kernel_size, strides, padding,
                             kernel_initializer = initializer)(x)
        return x
    return f

def Conv2DTranspose(in_channels, out_channels, kernel_size, strides = (2, 2),
                    padding = 'same', kernel_initializer = None):
    def f(x):
        if kernel_initializer is 'he_normal':
            kernel_shape = list(kernel_size) + [in_channels, out_channels]
            initializer = he_normal(kernel_shape)
        elif kernel_initializer is None:
            initializer = None
        else:
            raise Exception('invalid argument for kernel initializer')
        
        x = tf.layers.Conv2DTranspose(out_channels, kernel_size, strides, padding,
                                      kernel_initializer = initializer)(x)
        return x
    return f

def SubpixelConv(in_channels, out_channels, kernel_size = (1, 1), kernel_initializer = None):
    def f(x):
        x = Conv2D(in_channels, 4*out_channels, kernel_size,
                   kernel_initializer = kernel_initializer)(x)
        x = tf.depth_to_space(x, 2)
        return x
    return f

def MeanpoolConv(in_channels, out_channels, kernel_size = (1, 1), kernel_initializer = None):
    def f(x):
        x = tf.add_n([x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]])/4.
        x = Conv2D(in_channels, out_channels, kernel_size,
                   kernel_initializer = kernel_initializer)(x)
        return x
    return f

def ConvMeanpool(in_channels, out_channels, kernel_size = (1, 1), kernel_initializer = None):
    def f(x):
        x = Conv2D(in_channels, out_channels, kernel_size,
                   kernel_initializer = kernel_initializer)(x)
        x = tf.add_n([x[:, ::2, ::2], x[:, 1::2, ::2], x[:, ::2, 1::2], x[:, 1::2, 1::2]])/4.
        return x
    return f

def UpsamplingConv(in_channels, out_channels, kernel_size = (1, 1), kernel_initializer = None):
    def f(x):
        x = tf.concat([x, x, x, x], axis = 3)
        x = tf.depth_to_space(x, 2)
        x = Conv2D(in_channels, out_channels, kernel_size,
                   kernel_initializer = kernel_initializer)(x)
        return x
    return f


class PlainBlock(object):
    def __init__(self, in_channels, out_channels, kernel_size,
                 batch_norm = True, resample = None, name = 'plain'):
        self.in_chs = in_channels
        self.out_chs = out_channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.resample = resample
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            if self.resample == 'down':
                shortcut_layer = MeanpoolConv
                conv_1 = functools.partial(Conv2D, in_channels = self.in_chs, out_channels = self.in_chs)
                conv_2 = functools.partial(ConvMeanpool, in_channels = self.in_chs, out_channels = self.out_chs)
            elif self.resample == 'up':
                shortcut_layer = UpsamplingConv
                conv_1 = functools.partial(UpsamplingConv, in_channels = self.in_chs, out_channels = self.out_chs)
                conv_2 = functools.partial(Conv2D, in_channels = self.out_chs, out_channels = self.out_chs)
            elif self.resample == None:
                shortcut_layer = Conv2D
                conv_1 = functools.partial(Conv2D, in_channels = self.in_chs, out_channels = self.in_chs)
                conv_2 = functools.partial(Conv2D, in_channels = self.in_chs, out_channels = self.out_chs)
            else:
                raise Exception('invalid resample value')

            if self.out_chs == self.in_chs and self.resample == None:
                shortcut = x
            else:
                shortcut = shortcut_layer(in_channels = self.in_chs, out_channels = self.out_chs,
                                          kernel_size = (1, 1))(x)

            if self.batch_norm: x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
            x = conv_1(kernel_size = self.kernel_size, kernel_initializer = 'he_normal')(x)
            if self.batch_norm: x = tf.layers.BatchNormalization()(x)
            x = tf.nn.relu(x)
            x = conv_2(kernel_size = self.kernel_size, kernel_initializer = 'he_normal')(x)

            return shortcut + x


class BottleneckBlock(object):
    def __init__(self, in_channels, out_channels, kernel_size,
                 batch_norm = True, resample = None, name = 'bottleneck'):
        self.in_chs = in_channels
        self.out_chs = out_channels
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.resample = resample
        self.name = name

    def __call__(self, x, reuse = True):
        with tf.variable_scope(self.name) as vs:
            if reuse:
                vs.reuse_variables()

            '''resample: None, 'up', 'down' '''
            if self.resample == 'down':
                shortcut_layer = functools.partial(Conv2D, strides = (2, 2))
                conv_1 = functools.partial(Conv2D, in_channels = self.in_chs, out_channels = self.in_chs/2)
                conv_b = functools.partial(Conv2D, in_channels = self.in_chs/2, out_channels = self.out_chs/2,
                                           strides = (2, 2))
                conv_2 = functools.partial(Conv2D, in_channels = self.out_chs/2, out_channels = self.out_chs)
            elif self.resample == 'up':
                shortcut_layer = SubpixelConv
                conv_1 = functools.partial(Conv2D, in_channels = self.in_chs, out_channels = self.in_chs/2)
                conv_b = functools.partial(Conv2DTranspose, in_channels = self.in_chs/2, out_channels = self.out_chs/2)
                conv_2 = functools.partial(Conv2D, in_channels = self.out_chs/2, out_channels = self.out_chs)
            elif self.resample == None:
                shortcut_layer = Conv2D
                conv_1 = functools.partial(Conv2D, in_channels = self.in_chs, out_channels = self.in_chs/2)
                conv_b = functools.partial(Conv2D, in_channels = self.in_chs/2, out_channels = self.out_chs/2)
                conv_2 = functools.partial(Conv2D, in_channels = self.out_chs/2, out_channels = self.out_chs)

            else:
                raise Exception('invalid resample value')

            if self.out_chs == self.in_chs and self.resample == None:
                shortcut = x
            else:
                shortcut = shortcut_layer(in_channels = self.in_chs, out_channels = self.out_chs,
                                          kernel_size = (1, 1))(x)

            x = tf.nn.relu(x)
            x = conv_1(kernel_size = (1, 1), kernel_initializer = 'he_normal')(x)
            x = tf.nn.relu(x)
            x = conv_b(kernel_size = self.kernel_size, kernel_initializer = 'he_normal')(x)
            x = tf.nn.relu(x)
            x = conv_2(kernel_size = (1, 1), kernel_initializer = 'he_normal')(x)
            if self.batch_norm: x = tf.layers.BatchNormalization()(x)

            return shortcut + (0.3*x)


class ToRGB(object):
    def __init__(self, name = 'torgb'):
        self.name = name

    def __call__(self, x):
        with tf.variable_scope(self.name) as vs:
            x = tf.layers.Conv2D(3, (1, 1), padding = 'same')(x)
            return tf.nn.tanh(x)

