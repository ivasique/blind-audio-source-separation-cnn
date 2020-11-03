from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import keras.initializers

class BSS_CNN:
    @staticmethod
    def define(freq_bins, length):
        model = Sequential()

        inputShape = (freq_bins, length, 1)
        chanDim = -1
        if K.image_data_format() == "channels_first":
            inputShape = (1, freq_bins, length)
            chanDim = 1

        model.add(Conv2D(32, (3,3),
                         padding='same',
                         input_shape=inputShape,
                         kernel_initializer=keras.initializers.he_normal(seed=None),
                         bias_initializer='zeros'))
        model.add(LeakyReLU())
        model.add(BatchNormalization(axis=chanDim))        
        model.add(Conv2D(32, (3,3),
                         padding='same',
                         kernel_initializer=keras.initializers.he_normal(seed=None),
                         bias_initializer='zeros'))
        model.add(LeakyReLU())
        model.add(BatchNormalization(axis=chanDim))        
        model.add(MaxPooling2D(pool_size=(3,3)))
        model.add(Dropout(0.25))


        model.add(Conv2D(64, (3, 3),
                         padding='same',
                         kernel_initializer=keras.initializers.he_normal(seed=None),
                         bias_initializer='zeros'))
        model.add(LeakyReLU())
        model.add(BatchNormalization(axis=chanDim))       
        model.add(Conv2D(64, (3, 3),
                         padding='same',
                         kernel_initializer=keras.initializers.he_normal(seed=None),
                         bias_initializer='zeros'))
        model.add(LeakyReLU())
        model.add(BatchNormalization(axis=chanDim))      
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))       


        model.add(Conv2D(128, (3, 3),
                         padding='same',
                         kernel_initializer=keras.initializers.he_normal(seed=None),
                         bias_initializer='zeros'))
        model.add(LeakyReLU())
        model.add(BatchNormalization(axis=chanDim)) 
        model.add(Conv2D(128, (3, 3),
                         padding='same',
                         kernel_initializer=keras.initializers.he_normal(seed=None),
                         bias_initializer='zeros'))
        model.add(LeakyReLU())
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))


        model.add(Flatten())
        model.add(Dense(freq_bins*2))
        model.add(LeakyReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(freq_bins))
        model.add(Activation('sigmoid'))

        model.compile(loss='mean_squared_error',
                      optimizer='adam')

        return model