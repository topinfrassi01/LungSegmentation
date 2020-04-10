from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input, Add
from keras.layers import concatenate, core, Dropout
from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K

def _bottleneck_layer(inputs, filter_size):
    conv = Sequential([
        Conv2D(filter_size, 3, padding='same', activation='relu'),
        Conv2D(filter_size, 3, padding='same', activation='relu'),
        Conv2D(filter_size * 4, 1, padding='same')])(inputs)

    bottleneck = Conv2D(filter_size*4, 1, padding='same')

    return Activation(activation='relu')(Add()([bottleneck, conv]))


def create_dcan(input_size=(256,256,1)):
    # Encoder Path
    inputs = Input(input_size)

    conv1 = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    conv2 = Conv2D(32, 3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv2)

    bn1 = _bottleneck_layer(pool1, 64)
    bn2 = _bottleneck_layer(bn1, 64)
    pool2 = MaxPooling2D(pool_size=(2,2))(bn2)

    bn3 = _bottleneck_layer(pool2, 128)
    bn4 = _bottleneck_layer(bn3, 128)
    pool3 = MaxPooling2D(pool_size=(2,2))(bn4)

    bn5 = _bottleneck_layer(pool3, 128)
    bn6 = _bottleneck_layer(bn5, 128)
    drop1 = Dropout(0.5)(bn6)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop1)

    bn7 = _bottleneck_layer(drop1, 128)
    bn8 = _bottleneck_layer(bn7, 128)
    drop2 = Dropout(0.5)(bn8)    
    pool5 = MaxPooling2D(pool_size=(2,2))(drop2)

    bn9 = _bottleneck_layer(pool5, 128)
    bn10 = _bottleneck_layer(bn9, 128)
    drop2 = Dropout(0.5)(bn10)       

    # Decoder segmentation
    

    

    