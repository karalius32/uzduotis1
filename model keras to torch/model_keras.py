import keras
from keras.layers import Input


def Unet(shape=(224, 224, 1), num_of_classes=1, filters=8, depth=3, activation='relu', activation_args=None):
    inputs = Input(shape=shape)

    skip_connections = []

    enc_res, enc = EncodingBlock(input=inputs, filters=filters, kernel_size=5, max_pool=True, max_pool_size=2,
                                 batch_norm=True, activation=activation, activation_args=activation_args)
    skip_connections.append(enc_res)

    for i in range(depth):
        enc_res, enc = EncodingResBlock(input=enc, filters=filters * (2 ** (i + 1)), kernel_size=3,
										   max_pool=True, max_pool_size=2, batch_norm=True,
										   activation=activation, activation_args=activation_args)
        skip_connections.append(enc_res)

    _, bottleneck = EncodingResBlock(input=enc, filters=filters * (2 ** (depth + 1)), kernel_size=3,
									    max_pool=False, batch_norm=True, activation=activation,
									    activation_args=activation_args)
    bottleneck = AtrousSpatialPyramidPool(input=bottleneck, filters=filters * (2 ** (depth + 1)), kernel_size=3,
										  depth=3, activation=activation, activation_args=activation_args)

    dec = bottleneck
    for i in range(depth):
        dec = DecodingResBlock(input=dec, skip_connection=skip_connections[depth - i],
								  filters=filters * (2 ** (depth - i)), kernel_size=3, up_sample_size=2,
								  batch_norm=True, activation=activation, activation_args=activation_args)

    dec = DecodingResBlock(input=dec, skip_connection=skip_connections[0], filters=filters, kernel_size=3,
							  up_sample_size=2, batch_norm=True, activation=activation,
							  activation_args=activation_args)

    final_filter = num_of_classes
    final_activation = 'softmax' if final_filter > 1 else 'sigmoid'
    dec = ConvBlock(input=dec, filters=final_filter * 2, kernel_size=3, batch_norm=True, activation=activation,
				    activation_args=activation_args)
    outputs = Conv2D(filters=final_filter, kernel_size=(1, 1), activation=final_activation,
				     kernel_initializer='glorot_normal')(dec)

    model = Model(inputs, outputs)

    return model
	
	
def ActivationBlock(input, activation=None, activation_args=None):
    if activation is not None:
        if activation == 'leaky_relu':
            flow = LeakyReLU(alpha=activation_args['alpha'])(input)
        elif activation_args is None:
            flow = Activation(activation=activation)(input)
        else:
            flow = Activation(activation=activation, **activation_args)(input)
    else:
        flow = input
    return flow


def ConvBlock(input, filters, kernel_size=3, batch_norm=True, activation=None, activation_args=None):
    flow = ZeroPadding2D(padding=(kernel_size // 2, kernel_size // 2))(input)
    flow = Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), kernel_initializer='he_normal')(flow)
    if batch_norm:
        flow = BatchNormalization()(flow)
    flow = ActivationBlock(input=flow, activation=activation, activation_args=activation_args)
    return flow


def ResBlock(input, filters, kernel_size=3, batch_norm=True, activation=None, activation_args=None):
    shortcut = Conv2D(filters, kernel_size=(1, 1), kernel_initializer='he_normal')(input)
    flow = ConvBlock(input=input, filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    flow = ConvBlock(input=flow, filters=filters, kernel_size=kernel_size, batch_norm=batch_norm)
    flow = Add()([flow, shortcut])
    flow = ActivationBlock(input=flow, activation=activation, activation_args=activation_args)
    return flow
	
	
def EncodingBlock(input, filters=8, kernel_size=3, max_pool=True, max_pool_size=2, batch_norm=True, activation='relu', activation_args=None):
    flow = ConvBlock(input=input, filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    flow = ConvBlock(input=flow, filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    res = flow
    if max_pool:
        flow = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(flow)
    return res, flow
	
	
def EncodingResBlock(input, filters=8, kernel_size=3, max_pool=True, max_pool_size=2, batch_norm=True, activation='relu', activation_args=None):
    flow = ResBlock(input=input, filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    res = flow
    if max_pool:
        flow = MaxPooling2D(pool_size=(max_pool_size, max_pool_size))(flow)
    return res, flow
	
	
def DecodingResBlock(input, skip_connection, filters=8, kernel_size=3, up_sample_size=2, batch_norm=True, activation='relu', activation_args=None):
    flow = UpSampling2D(size=(up_sample_size, up_sample_size), interpolation='bilinear')(input)
    flow = ConvBlock(input=flow, filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    flow = Concatenate(axis=3)([flow, skip_connection])
    flow = ResBlock(input=flow, filters=filters, kernel_size=kernel_size, batch_norm=batch_norm, activation=activation, activation_args=activation_args)
    return flow
	
	
def AtrousSpatialPyramidPool(input, filters=8, kernel_size=3, depth=3, activation='relu', activation_args=None):
    dilations = []
    for i in range(depth):
        dilation_rate = 2 ** i
        dilation = ZeroPadding2D(padding=(dilation_rate * (kernel_size // 2), dilation_rate * (kernel_size // 2)))(input)
        dilation = Conv2D(filters, kernel_size, dilation_rate=dilation_rate, kernel_initializer='he_normal')(dilation)
        dilation = BatchNormalization()(dilation)
        dilation = ActivationBlock(input=dilation, activation=activation, activation_args=activation_args)
        dilations.append(dilation)

    h, w = input.shape[1:3]
    pool = AveragePooling2D(pool_size=(h, w))(input)
    pool = UpSampling2D(size=(h, w), interpolation='bilinear')(pool)

    flow = Concatenate()([*dilations, pool])
    flow = ConvBlock(input=flow, filters=filters, kernel_size=1, batch_norm=True, activation=activation, activation_args=activation_args)
    return flow