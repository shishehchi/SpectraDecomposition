import numpy as np
from glob import glob
import astropy.units as u
import tensorflow 
from tensorflow.keras.layers import Input,concatenate, Dense, UpSampling1D,Conv1D,Flatten,Reshape,Dropout,LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
backend.set_image_data_format('channels_last')

def Model_class(reg=.00,use_bias=False):
    f_size=4
    spect_dim=2208
    dim_out=144

    

    input_img = Input(shape=(spect_dim,1))  # adapt this if using `channels_first` image data format

    x = Conv1D(64, 11, strides=2, padding='same')(input_img)
    x= LeakyReLU()(x)

    x = Conv1D(64+32,9 ,strides= 2, padding='same', use_bias=use_bias,kernel_regularizer =tf.keras.regularizers.l2( l=reg))(x)
    x= LeakyReLU()(x)
    
    x = Conv1D(128, 7,strides= 2, padding='same', use_bias=use_bias,kernel_regularizer =tf.keras.regularizers.l2( l=reg))(x)
    x= LeakyReLU()(x)

    x = Conv1D(128+32, 7,strides= 2, padding='same', use_bias=use_bias,kernel_regularizer =tf.keras.regularizers.l2( l=reg))(x)
    x= LeakyReLU()(x)

    x = Conv1D(128+64, 5,strides= 2, padding='same', use_bias=use_bias,kernel_regularizer =tf.keras.regularizers.l2( l=reg))(x)
    x= LeakyReLU()(x)

        
    x = Conv1D(256, 5,strides= 2, padding='same', use_bias=use_bias,kernel_regularizer =tf.keras.regularizers.l2( l=reg))(x)
    x= LeakyReLU()(x)

    x = Conv1D(256+64, 5,strides= 2, padding='same', use_bias=use_bias,kernel_regularizer =tf.keras.regularizers.l2( l=reg))(x)
    x= LeakyReLU()(x)

    x = Conv1D(256+96,5, strides= 1, padding='same', use_bias=use_bias,kernel_regularizer =tf.keras.regularizers.l2( l=reg))(x)
    x= LeakyReLU()(x)
    
    shape_before_flattening = K.int_shape(x)[1:]

    w, h = shape_before_flattening
    num_neurons = w*h

    x = Flatten(name = 'flat')(x)

    x = Dense(256+256)(x)
    x= LeakyReLU()(x)
    x= Dropout(0.25)(x)
    
    x = Dense(512)(x)
    x= LeakyReLU()(x)
    x= Dropout(0.25)(x)
    
    x = Dense(512+256)(x)
    x= LeakyReLU()(x)
    x= Dropout(0.35)(x)

    output_agn = Dense(1024)(x)
    output_agn= LeakyReLU()(output_agn)
    x_agn= Dropout(0.3)(output_agn)
    output_agn = Dense(spect_dim,activation='linear',name='AGN')(x_agn)
    

    output_sf = Dense(1024)(x)
    output_sf= LeakyReLU()(output_sf)
    x_sf= Dropout(0.3)(output_sf)
    output_sf = Dense(spect_dim,activation='linear',name='SF')(x_sf)

    
    concat_agn_ssf = concatenate([x_sf,x_agn])
    
    output_norm= Dense(100)(concat_agn_ssf)
    output_norm= LeakyReLU()(output_norm)
    output_norm= Dropout(0.3)(output_norm)
    output_norm= Dense(10)(output_norm)
    output_norm= LeakyReLU()(output_norm)
    output_norm= Dropout(0.3)(output_norm)
    output_norm= Dense(1,name='SF_Prob')(output_norm)

    output_norm_agn= Dense(100)(concat_agn_ssf)
    output_norm_agn= LeakyReLU()(output_norm_agn)
    output_norm_agn= Dropout(0.3)(output_norm_agn)
    output_norm_agn= Dense(10)(output_norm_agn)
    output_norm_agn= LeakyReLU()(output_norm_agn)
    output_norm_agn= Dropout(0.3)(output_norm_agn)
    output_norm_agn= Dense(1,name='AGN_Prob')(output_norm_agn)
    
        
    dadded_sf = tf.keras.layers.Add()([output_norm, output_norm_agn])
    dadded_sf= LeakyReLU()(dadded_sf)
    dadded_sf= Dense(1,name='Tot_Prob')(dadded_sf)    

    
    output_SF= Dense(100)(output_sf)
    output_SF= LeakyReLU()(output_SF)
    output_SF= Dense(10)(output_SF)
    output_SF= LeakyReLU()(output_SF)
    mass_SF= Dense(1,name='MASS_SF')(output_SF)

    
    output_AGN= Dense(100)(output_agn)
    output_AGN= LeakyReLU()(output_AGN)
    output_AGN= Dense(10)(output_AGN)
    output_AGN= LeakyReLU()(output_AGN)
    mass_AGN= Dense(1,name='MASS_AGN')(output_AGN)
    
    model = Model(input_img, [output_sf,output_agn,output_norm,output_norm_agn,dadded_sf])   
    model.compile(loss=['mse','mse','mse','mse','mse'],  metrics=['mse'],loss_weights=[1,1,1,1,1])
    model.load_weights('WeightCLSm_NFSP_TO_ONE8.hdf5')

    
    return model
