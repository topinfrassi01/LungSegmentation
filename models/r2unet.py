from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, Input, Add, Concatenate, Dropout, Layer
from keras.models import Model
from keras.optimizers import Adam
from loss import dice_coef, jaccard_distance

class ResidualConv2DBlock(Layer):
    def __init__(self, n_filters, t, kernel_size=3, stride=1, padding="same", activation="relu", use_batch_norm=False, data_format="channels_last"):
        self.t = t
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        self.convolution_layers = []
        if use_batch_norm:
            self.batch_normalizations = []

        for i in range(self.t):
            self.convolution_layers.append(Conv2D(filters=n_filters, kernel_size=kernel_size, strides=1, activation="None", padding=padding, data_format=data_format))

            if use_batch_norm:
                self.batch_normalizations.append(BatchNormalization())

    def call(self, inputs):
        for i in range(self.t):
            convolution = self.convolution_layers[i]
            
            if i == 0:
                result = convolution(inputs)
            else :
                result = convolution(result)

            if self.use_batch_norm:
                batch_normalisation = self.batch_normalizations[i]
                result = batch_normalisation(result)

            result = Activation(self.activation)(result)

        return Add()([inputs, result])

class RecurrentResidualConv2DBlock(Layer):
    def __init__(self, t_recurrent, t_residual, n_filters, kernel_size=3, stride=1, padding="same", activation="relu", use_batch_norm=False, data_format="channels_last"):
        self.t_recurrent = t_recurrent
        self.t_residual = t_residual

        self.residual_blocks = []
        for i in range(t_recurrent):
            self.residual_blocks.append(ResidualConv2DBlock(t_residual, n_filters, kernel_size, stride, padding, activation, use_batch_norm, data_format))
    
    def call(self, inputs):
        for i in range(self.t_recurrent):
            res_block = self.residual_blocks[i]

            if i == 0:
                result = res_block(inputs)
            else:
                result = res_block(result)

        return Add()([inputs, result])


def build_r2_unet(image_shape=(256,256,1), n_classes=2, lr=1e-6, t_residual=2, t_recurrent=2):
    inputs = Input(image_shape)
    x = inputs
    depth = 3
    features = 64
    skips = []
    for i in range(depth):
        x = RecurrentResidualConv2DBlock(t_recurrent=t_recurrent, t_residual=t_residual, n_filters=features)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

        features = features * 2

    x = RecurrentResidualConv2DBlock(t_recurrent=t_recurrent, t_residual=t_residual, n_filters=features)(x)

    for i in reversed(range(depth)):
        features = features // 2
        up = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=3)([up, skips[i]])
        x = RecurrentResidualConv2DBlock(t_recurrent=t_recurrent, t_residual=t_residual, n_filters=features)(x)

    conv6 = Conv2D(n_classes, 1, padding='same', activation="softmax")(x)

    model = Model(inputs=inputs, outputs=conv6)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=['accuracy', dice_coef, jaccard_distance])

    model.summary()
    
    return model