# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 22:47:04 2018

@author: Manoochehr
"""
from keras.layers import  Conv1D, MaxPooling1D, Flatten, Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.layers import Input
from keras.models import Model

# ///////////////////////////// Genuine /////////////////////////////////////////////////
def build_category_branch(inputs, numLastLayer, finalAct, LayerName):
        x = Conv1D(96, 11, strides=4, padding= 'valid', kernel_regularizer = regularizers.l2(0.01))(inputs)
        x = BatchNormalization()(x)
        x = Activation( activation='relu')(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        x = Conv1D(256, 5, strides=1, padding= 'same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation( activation='relu')(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        x = Conv1D(384, 3, strides=1, padding= 'same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation( activation='relu')(x)
        x = Conv1D(384, 3, strides=1, padding= 'same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation( activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Conv1D(256, 3, strides=1, padding= 'same', kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation( activation='relu')(x)
        x = MaxPooling1D(3, strides=2, padding='same')(x)
        x = Flatten()(x)
        x = Dense(units = 2048 ,kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation(activation='relu')(x)
        x = Dense(units = 2048, kernel_regularizer=regularizers.l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Activation( activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(units = numLastLayer, kernel_regularizer=regularizers.l2(0.01))(x)
        x = Activation(finalAct, name = LayerName)(x)
        
        return x


def build(width, height, numUsers, numForgOrGen, finalAct1, finalAct2):
        inputShape = (width, height)
		        
        # construct both the "users" and "GenOrForg" sub-networks
        inputs = Input(shape=inputShape)
        
        categoryUser = build_category_branch(inputs, numUsers, finalAct1, LayerName="categoryUser")
        
        categoryGenOrForg = build_category_branch(inputs, numForgOrGen, finalAct2, LayerName="categoryGenOrForg")
        
        model = Model(
			inputs=inputs,
			outputs=[categoryUser, categoryGenOrForg],
			name="SignatureNet")        
        # return the constructed network architecture
        return model

