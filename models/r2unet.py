from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input, Add, Concatenate, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from loss import dice_coef, jaccard_distance

class ResidualConv2DBlock(Layer):
    def __init__(self, n_filters, depth=3, kernel_size=3, stride=3, padding="same", activation="relu", use_batch_norm=False, data_format="channels_last"):
        self.depth = depth
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.n_filters = n_filters
        self.convolution_layers = []

        if use_batch_norm:
            self.batch_normalizations = []

        self.convolution_layers.append(Conv2D(filters=n_filters, kernel_size=1, strides=1, activation="None", padding=padding, data_format=data_format))

        for i in range(self.depth - 2):
            self.convolution_layers.append(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=stride, activation="None", padding=padding, data_format=data_format))

            if use_batch_norm:
                #To account for the first and last layer
                if i == 0:
                    self.batch_normalizations.append(BatchNormalization())    
                    self.batch_normalizations.append(BatchNormalization())

                self.batch_normalizations.append(BatchNormalization())

        self.convolution_layers.append(Conv2D(filters=n_filters, kernel_size=1, strides=1, activation="None", padding=padding, data_format=data_format))

        self.shortcut_conv = Conv2D(filters=n_filters, kernel_size=1, strides=1, activation="relu", padding=padding, data_format=data_format)

    def call(self, inputs):
        result = inputs
        i = 0
        for convolution in self.convolution_layers:
            result = convolution(inputs)

            if self.use_batch_norm:
                batch_normalisation = self.batch_normalizations[i]
                result = batch_normalisation(result)

            result = Activation(self.activation)(result)
            i += 1

        if not inputs.shape[3] == self.n_filters:
            inputs = self.shortcut_conv(inputs)

        return Add()([inputs, result])

class RecurrentResidualConv2DBlock(Layer):
    def __init__(self, t_recurrent, depth, n_filters, kernel_size=3, stride=1, padding="same", activation="relu", use_batch_norm=False, data_format="channels_last"):
        self.t_recurrent = t_recurrent
        self.depth = depth

        self.residual_blocks = []
        for i in range(t_recurrent):
            self.residual_blocks.append(ResidualConv2DBlock(depth, n_filters, kernel_size, stride, padding, activation, use_batch_norm, data_format))
    
    def call(self, inputs):
        for i in range(self.t_recurrent):
            res_block = self.residual_blocks[i]

            if i == 0:
                result = res_block(inputs)
            else:
                result = res_block(result)

        return Add()([inputs, result])


def build_r2unet(image_shape=(256,256,1), n_classes=2, lr=1e-6, depth=2, t_recurrent=2):
    inputs = Input(image_shape)
    x = inputs
    depth = 3
    features = 64
    skips = []
    for i in range(depth):
        x = RecurrentResidualConv2DBlock(t_recurrent=t_recurrent, depth=depth, n_filters=features)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

        features = features * 2

    x = RecurrentResidualConv2DBlock(t_recurrent=t_recurrent, depth=depth, n_filters=features)(x)

    for i in reversed(range(depth)):
        features = features // 2
        up = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=3)([up, skips[i]])
        x = RecurrentResidualConv2DBlock(t_recurrent=t_recurrent, depth=depth, n_filters=features)(x)

    conv6 = Conv2D(n_classes, 1, padding='same', activation="softmax")(x)

    model = Model(inputs, conv6)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=['accuracy', dice_coef, jaccard_distance])

    model.summary()
    
    return model