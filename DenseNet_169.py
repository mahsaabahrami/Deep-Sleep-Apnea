# =============================================================================
# Import Libraries:
import keras
from tensorflow.keras import Model 
import tensorflow.keras.backend as K
# =============================================================================

def Batch_Conv(x, kn, ks=1, s=1):
    
        '''
        This block applies BatchNormalization -> Relu -> Convolution 
            Parameters:
                ----------
                x : input vector
                kn : number of kernels 
                ks : kernel size
                s : stride
                
        '''
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU()(x)
        x = keras.layers.Conv2D(kn, (ks, 1), strides=(s, 1), padding = 'same')(x)
        
        return x
# =============================================================================

def Dense_Block(x, repetition):

    filters = 32   
    for _ in range(repetition):
            y = Batch_Conv(x, 4*filters)
            y = Batch_Conv(y, filters, 3)
            x = keras.layers.concatenate([y,x])
    return x
# =============================================================================

def transition_layer(x):
        
        x = Batch_Conv(x, K.int_shape(x)[-1] //2 )
        x = keras.layers.AvgPool2D((2,1), strides = (2,1), padding = 'same')(x)
        
        return x

# =============================================================================

def DenseNet_169(input_shape, n_classes, filters = 32):

    input = keras.layers.Input (input_shape)
    
    x = keras.layers.Conv2D(64, (7,1), strides = (2,1), padding = 'same')(input)
    x = keras.layers.MaxPool2D((3,1), strides = (2,1), padding = 'same')(x)
    
    
    for repetition in [6,12,32,16]:
        
        d = Dense_Block(x, repetition)
        x = transition_layer(d)
        
        
    x = keras.layers.GlobalAveragePooling2D()(d)
    output = keras.layers.Dense(n_classes, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model 

# =============================================================================

def DenseNet_169_Tiny(input_shape, n_classes, filters = 32):

    input = keras.layers.Input (input_shape)
    
    x = keras.layers.Conv2D(64, (7,1), strides = (2,1), padding = 'same')(input)
    x = keras.layers.MaxPool2D((3,1), strides = (2,1), padding = 'same')(x)
    
    
    for repetition in [6,12,24,24]:
        
        d = Dense_Block(x, repetition)
        x = transition_layer(d)
        
        
    x = keras.layers.GlobalAveragePooling2D()(d)
    output = keras.layers.Dense(n_classes, activation = 'softmax')(x)
    
    model = Model(input, output)
    return model 

# =============================================================================

model = DenseNet_169_Tiny(input_shape=(6000, 1, 1), n_classes=2, filters = 32)
model.summary()