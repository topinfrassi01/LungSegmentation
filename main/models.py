from tensorflow.keras.layers import Input, Conv2D, Layer, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Activation, Concatenate, Conv2D, MaxPooling2D, Dropout, \
    Concatenate, UpSampling2D, Input, BatchNormalization, Activation, Input, Add, Concatenate, Dropout, Layer
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from loss import dice_coef, jaccard_distance
from tensorflow.keras.regularizers import l2

class Conv2DWithBatchNorm(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, use_batch_norm=True, padding="same", activation="relu", name=None):
        super(Conv2DWithBatchNorm, self).__init__(name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_batch_norm = use_batch_norm
        self.padding = padding
        self.activation = activation
        
        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=None)

        if self.use_batch_norm:
            self.batch_norm = BatchNormalization()

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            'padding' : self.padding,
            'activation' : self.activation,
            'use_batch_norm' : self.use_batch_norm,
            'kernel_size' : self.kernel_size,
            'strides' : self.strides
        })
        return config

    def call(self, inputs):
        result = self.conv2d(inputs)

        if self.use_batch_norm:
            result = self.batch_norm(result)

        result = Activation(self.activation)(result)

        return result

def build_dcan(input_shape=(256,256,1), n_classes=3, lr=5e-4):
    inputs = Input(input_shape)

    conv1 = Conv2DWithBatchNorm(filters=64)(inputs)
    conv2 = Conv2DWithBatchNorm(filters=64)(conv1)
    mp1 = MaxPooling2D(pool_size=(2,2))(conv2)

    conv3 = Conv2DWithBatchNorm(filters=128)(mp1)
    conv4 = Conv2DWithBatchNorm(filters=128)(conv3)
    mp2 = MaxPooling2D(pool_size=(2,2))(conv4)

    conv5 = Conv2DWithBatchNorm(filters=256)(mp2)
    conv6 = Conv2DWithBatchNorm(filters=256)(conv5)
    mp3 = MaxPooling2D(pool_size=(2,2))(conv6)

    conv7 = Conv2DWithBatchNorm(filters=512)(mp3)
    conv8 = Conv2DWithBatchNorm(filters=512)(conv7)
    #conv8 size should be 8 times smaller than input
    mp4 = MaxPooling2D(pool_size=(2,2))(conv8)

    conv9 = Conv2DWithBatchNorm(filters=512)(mp4)
    conv10 = Conv2DWithBatchNorm(filters=512)(conv9)
    #conv10 size should be 16 times smaller than input
    mp5 = MaxPooling2D(pool_size=(2,2))(conv10)

    conv11 = Conv2DWithBatchNorm(filters=1024)(mp5)
    conv12 = Conv2DWithBatchNorm(filters=1024)(conv11)
    #conv12 size should be 32 times smaller than input

    # Region segmentation
    region_upsample_1 = Conv2DWithBatchNorm(64, name="conv2d_regions_upsample1")(UpSampling2D(size = (8,8))(conv8))
    region_upsample_1 = Dropout(0.5)(region_upsample_1)

    region_upsample_2 = Conv2DWithBatchNorm(64, name="conv2d_regions_upsample2")(UpSampling2D(size = (16,16))(conv10))
    region_upsample_2 = Dropout(0.5)(region_upsample_2)

    region_upsample_3 = Conv2DWithBatchNorm(64, name="conv2d_regions_upsample3")(UpSampling2D(size = (32,32))(conv12))
    region_upsample_3 = Dropout(0.5)(region_upsample_3)

    region_output = Concatenate()([region_upsample_1, region_upsample_2, region_upsample_3])
    region_output = Conv2D(n_classes, 1, activation="softmax", name="conv2d_regions_output")(region_output)
    

    # Region segmentation
    contours_upsample_1 = Conv2DWithBatchNorm(64, name="conv2d_contours_upsample1")(UpSampling2D(size = (8,8))(conv8))
    contours_upsample_1 = Dropout(0.5)(contours_upsample_1)

    contours_upsample_2 = Conv2DWithBatchNorm(64, name="conv2d_contours_upsample2")(UpSampling2D(size = (16,16))(conv10))
    contours_upsample_2 = Dropout(0.5)(contours_upsample_2)

    contours_upsample_3 = Conv2DWithBatchNorm(64, name="conv2d_contours_upsample3")(UpSampling2D(size = (32,32))(conv12))
    contours_upsample_3 = Dropout(0.5)(contours_upsample_3)

    contours_output = Concatenate()([contours_upsample_1, contours_upsample_2, contours_upsample_3])
    contours_output = Conv2D(n_classes, 1, activation="softmax", name="contours_output")(contours_output)
    
    model = Model(inputs, [region_output, contours_output])
    model.compile(optimizer=Adam(lr=lr), loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=['accuracy', dice_coef, jaccard_distance])
    
    model.summary()
    
    return model

class RecurrentConv2D(Layer):
    def __init__(self, filters, t, kernel_size=3, strides=1, use_batch_norm=True, activation='relu'):
        super(RecurrentConv2D, self).__init__()
        self.filters = filters
        self.t = t
        self.use_batch_norm = use_batch_norm
        self.activation = activation
        self.kernel_size = kernel_size
        self.strides = strides

        self.conv_layers = []
        self.batch_norm_layers = []
        self.activation_layer = Activation(activation)

        self.shortcut_conv = Conv2D(filters, kernel_size=1, strides=1, activation=None, padding='same')

        for _ in range(self.t + 1):
            self.conv_layers.append(Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=None, padding='same'))
            self.batch_norm_layers.append(BatchNormalization())

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters' : self.filters,
            't' : self.t,
            'activation' : self.activation,
            'use_batch_norm' : self.use_batch_norm,
            'kernel_size' : self.kernel_size,
            'strides' : self.strides
        })
        return config

    def call(self, inputs):
        x = inputs

        if not inputs.shape[3] == self.filters:
            inputs = self.shortcut_conv(inputs)

        for i in range(self.t + 1):
            conv = self.conv_layers[i]
            batch_norm = self.batch_norm_layers[i]

            x = conv(x)
            x = batch_norm(x)
            x = self.activation_layer(x)

            x = Add()([x, inputs])

        return x

class ResidualRecurrentConv2D(Layer):
    def __init__(self, filters_in, filters_out, t, use_batch_norm=True, activation="relu"):
        super(ResidualRecurrentConv2D, self).__init__()
        self.filters_in = filters_in
        self.filters_out = filters_out
        self.activation = activation
        self.use_batch_norm = use_batch_norm

        self.convolution_layers = [] 
        self.batch_norm_layers = []
        self.activation_layer = Activation(activation)

        self.shortcut_conv = RecurrentConv2D(filters_out, t=3, kernel_size=1, strides=1) #kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))
                
        self.convolution_layers.append(RecurrentConv2D(filters_in, t=3, kernel_size=1, strides=1))
        self.convolution_layers.append(RecurrentConv2D(filters_in, t=3, kernel_size=3, strides=1))
        self.convolution_layers.append(RecurrentConv2D(filters_out, t=3, kernel_size=1, strides=1))

        if use_batch_norm:
            self.batch_norm_layers.append(BatchNormalization())
            self.batch_norm_layers.append(BatchNormalization())
            self.batch_norm_layers.append(BatchNormalization())

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters_in' : self.filters_in,
            'filters_out' : self.filters_out,
            'activation' : self.activation,
            'use_batch_norm' : self.use_batch_norm
        })
        return config

    def call(self, inputs):
        x = inputs
        for i in range(3):
            conv = self.convolution_layers[i]
            x = conv(x)

            if self.use_batch_norm:
                batch_norm = self.batch_norm_layers[i]
                x = batch_norm(x)

            x = self.activation_layer(x)

        if not inputs.shape[3] == self.filters_out:
            inputs = self.shortcut_conv(inputs)

        return Add()([x, inputs])
        
def build_r2unet(image_shape=(256,256,1), depth=4, n_classes=3, lr=1e-6, t=3):
    inputs = Input(image_shape)
    x = inputs
    filters_in = 1
    filters_out = 64
    skips = []
    for i in range(depth):
        x = ResidualRecurrentConv2D(filters_in=filters_in, filters_out=filters_out, t=t)(x)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

        filters_in = filters_out
        filters_out = filters_out * 2

    x = ResidualRecurrentConv2D(filters_in=filters_in, filters_out=filters_out, t=t)(x)

    temp = filters_in
    filters_in = filters_out
    filters_out = temp

    for i in reversed(range(depth)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate(axis=3)([x, skips[i]])
        x = ResidualRecurrentConv2D(filters_in=filters_in, filters_out=filters_out, t=t)(x)

        filters_in = filters_out
        filters_out = filters_out / 2
    
    conv6 = Conv2D(n_classes, kernel_size=1, strides=1, padding='same', activation="softmax")(x)

    model = Model(inputs, conv6)

    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=['accuracy', dice_coef, jaccard_distance])

    model.summary()
    
    return model



def build_unet(input_size = (256,256,1), n_classes=3, lr=0.0001, momentum=0.99, dropout_rate=0.5):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = Concatenate(axis=3)([drop4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = Concatenate(axis=3)([conv3,up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = Concatenate(axis=3)([conv2,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = Concatenate(axis=3)([conv1,up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(n_classes, 1, activation = 'softmax')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer = SGD(lr=lr, momentum=momentum), loss = 'categorical_crossentropy', metrics = ['accuracy', dice_coef, jaccard_distance])
    
    #model.summary()

    return model