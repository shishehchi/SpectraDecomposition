import tensorflow 
from tensorflow.keras.layers import Input, Dense,Conv1D,Flatten,Dropout,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow.keras
import tensorflow as tf
from tensorflow.keras import backend
backend.set_image_data_format('channels_last')

def Flux_estimator():
    spect_dim=2208
    line_dim=100
    n_lines=8
    n_out1= 1024
    n_out2=1024

    input_spec = Input(shape=(spect_dim,1),name="input_spec")

    x1 = Conv1D(128, 10,  strides=2, padding='same',name='C1')(input_spec)
    x1= LeakyReLU()(x1)

    x1 = Conv1D(128+32, 8,strides= 2, padding='same',name='C2')(x1)
    x1= LeakyReLU()(x1)

    x1 = Conv1D(128+64, 6, strides=2, padding='same',name='C3')(x1)
    x1= LeakyReLU()(x1)

    x1 = Conv1D(128+96, 6, strides=2, padding='same',name='C4')(x1)
    x1= LeakyReLU()(x1)

    x1 = Conv1D(128+96, 6,strides=2, padding='same',name='C5')(x1)
    x1= LeakyReLU()(x1)

    x1 = Conv1D(128+128, 6, strides=2, padding='same',name='C7')(x1)
    x1= LeakyReLU()(x1)

    x1 = Conv1D(128+128, 6, strides=1, padding='same',name='C8')(x1)
    x1= LeakyReLU()(x1)

    x1 = Conv1D(256, 6, strides=1, padding='same',name='C9')(x1)
    x1= LeakyReLU()(x1)

    shape_before_flattening_1 = K.int_shape(x1)[1:]
    x1 = Flatten(name = 'flat1')(x1)

    x1=Dense(1024)(x1)
    x1= LeakyReLU()(x1)
    x1= LeakyReLU()(x1)

    x1=Dense(512)(x1)
    x1= LeakyReLU()(x1)
    x1= Dropout(0.25)(x1)

    output1=Dense(n_lines,activation='linear',name='Out_Ratio')(x1)

    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002, rho=0.9, momentum=0.005, epsilon=1e-07, centered=False, name="RMSprop")

    model = Model(input_spec, output1)
    model.compile(optimizer=optimizer, loss='mse')

    model.load_weights('Weight_REGms_BN29.hdf5')
    return model
