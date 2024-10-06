# =============================================================================
# import libraries:
import keras
from tensorflow.keras import Model
from keras.initializers import glorot_uniform
# =============================================================================

def Identitiy_Block(X, filters):
    '''
    Parameters
    ----------
    X : Input 
    filters : Number of Filters

    Returns
    -------
    X : output of Identity Blocks

    '''

    
    F2, F3 = filters
    
    X_shortcut = X
        
        
    X = keras.layers.Conv2D(filters = F2, kernel_size = (3, 1), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis = 3)(X)
    X = keras.layers.Activation('relu')(X)

    X = keras.layers.Conv2D(filters = F3, kernel_size = (3, 1), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis = 3)(X)


    X = keras.layers.Add()([X_shortcut, X])
    X = keras.layers.Activation('relu')(X)
        
    return X
# =============================================================================

def Convolution_Block(X, filters):

    '''
    Parameters
    ----------
    X : Input 
    filters : Number of Filters


    Returns
    -------
    X : output of Convolutional Blocks

    '''

    F1, F2 = filters

    X_shortcut = X

    X = keras.layers.Conv2D(filters=F1, kernel_size=(3, 1), strides=(1, 1), padding='same',  kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)


    X = keras.layers.Conv2D(filters=F2, kernel_size=(3, 1), strides=(1, 1), padding='same',  kernel_initializer=glorot_uniform(seed=0))(X)
    X = keras.layers.BatchNormalization(axis=3)(X)

    X_shortcut = keras.layers.Conv2D(filters=F2, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis=3)(X_shortcut)

    X = keras.layers.Add()([X, X_shortcut])
    X = keras.layers.Activation('relu')(X)

    return X
# =============================================================================

def ResNet18(input_shape=(6000,1, 1)):
    
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Conv2D(64, (7, 1), strides=(2, 1), kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D((3, 1), strides=(2, 1))(X)


    X = Convolution_Block(X, filters = [64, 64])
    X = Identitiy_Block(X, filters = [64, 64])

   

    X = Convolution_Block(X,  filters=[128, 128])
    X = Identitiy_Block( X,   filters=[128, 128])

    
    X = Convolution_Block(X,  filters=[256, 256])
    X = Identitiy_Block( X,   filters=[256, 256])

    
    X = Convolution_Block(X,  filters=[512, 512])
    X = Identitiy_Block( X,   filters=[512, 512])

    X = keras.layers.AveragePooling2D(pool_size=(2, 1), padding='same')(X)

    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(2, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X) 
    model = Model(inputs=X_input, outputs=X, name='ResNet34')

    return model
# =============================================================================
def ResNet18_tiny(input_shape=(6000,1, 1)):
    
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Conv2D(64, (7, 1), strides=(2, 1), kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = keras.layers.BatchNormalization(axis=3)(X)
    X = keras.layers.Activation('relu')(X)
    X = keras.layers.MaxPooling2D((3, 1), strides=(2, 1))(X)


    X = Convolution_Block(X, filters = [64, 64])
    X = Identitiy_Block( X,  filters = [64, 64])
   

    X = Convolution_Block(X,  filters=[128, 128])
    X = Identitiy_Block( X,   filters=[128, 128])

    
    X = Convolution_Block(X,  filters=[256, 256])
    X = Identitiy_Block( X,   filters=[256, 256])


    X = keras.layers.AveragePooling2D(pool_size=(2, 1), padding='same')(X)
    
    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(2, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(X) 
    model = Model(inputs=X_input, outputs=X, name='ResNet34')

    return model
# =============================================================================
model1=ResNet18_tiny(input_shape=(6000,1, 1))
model1.summary()

