from tensorflow.keras.layers import Input, Conv2D, Layer, BatchNormalization, MaxPooling2D, Dropout, UpSampling2D, Activation, Concatenate
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from loss import dice_coef, jaccard_distance

class Conv2DWithBatchNorm(Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding="same", activation="relu", data_format="channels_last"):
        self.conv2d = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, activation="None")
        self.batch_norm = BatchNormalization()
        self.activation = Activation(activation)

    def call(self, inputs):
        result = self.conv2d(inputs)
        result = self.batch_norm(result)
        result = self.activation(result)

        return result

def build_dcan(input_shape=(256,256,1), n_classes=2, lr=5e-4):
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
    region_upsample_1 = Conv2DWithBatchNorm(64)(UpSampling2D(size = (8,8))(conv8))
    region_upsample_1 = Dropout(0.5)(region_upsample_1)

    region_upsample_2 = Conv2DWithBatchNorm(64)(UpSampling2D(size = (16,16))(conv10))
    region_upsample_2 = Dropout(0.5)(region_upsample_2)

    region_upsample_3 = Conv2DWithBatchNorm(64)(UpSampling2D(size = (32,32))(conv12))
    region_upsample_3 = Dropout(0.5)(region_upsample_3)

    region_output = Concatenate()([region_upsample_1, region_upsample_2, region_upsample_3])
    region_output = Conv2D(n_classes, 1, activation="softmax")
    

    # Region segmentation
    contours_upsample_1 = Conv2DWithBatchNorm(64)(UpSampling2D(size = (8,8))(conv8))
    contours_upsample_1 = Dropout(0.5)(contours_upsample_1)

    contours_upsample_2 = Conv2DWithBatchNorm(64)(UpSampling2D(size = (16,16))(conv10))
    contours_upsample_2 = Dropout(0.5)(contours_upsample_2)

    contours_upsample_3 = Conv2DWithBatchNorm(64)(UpSampling2D(size = (32,32))(conv12))
    contours_upsample_3 = Dropout(0.5)(contours_upsample_3)

    contours_output = Concatenate()([contours_upsample_1, contours_upsample_2, contours_upsample_3])
    contours_output = Conv2D(n_classes, 1, activation="softmax")
    
    model = Model(inputs, [region_output, contours_output])
    model.compile(optimizer=Adam(lr=lr), loss=["categorical_crossentropy", "categorical_crossentropy"], metrics=['accuracy', dice_coef, jaccard_distance])
    
    return model