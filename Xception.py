# =============================================================================
# Import Libraries:

import keras
from tensorflow.keras import Model

# =============================================================================

def Conv_BN(x, filters, kernel_size, strides=1):
    
    x = keras.layers.Conv2D(filters=filters, kernel_size = kernel_size, strides=strides, padding = 'same',use_bias = False)(x)
    x = keras.layers.BatchNormalization()(x)
    
    return x

# =============================================================================

def Sep_BN(x, filters, kernel_size, strides=1):
    
    x = keras.layers.SeparableConv2D(filters=filters, kernel_size = kernel_size, 
                        strides=strides, 
                        padding = 'same', 
                        use_bias = False)(x)
    x = keras.layers.BatchNormalization()(x)
    return x
# =============================================================================

def Entry_Flow(x):
    
    x = Conv_BN(x, filters =32, kernel_size =(3,1), strides=(2,1))
    x = keras.layers.ReLU()(x)
    x = Conv_BN(x, filters =64, kernel_size =(3,1), strides=(1,1))
    tensor = keras.layers.ReLU()(x)
    
    x = Sep_BN(tensor, filters = 128, kernel_size =(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters = 128, kernel_size =(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding = 'same')(x)
    
    tensor = Conv_BN(tensor, filters=128, kernel_size = 1,strides=(2,1))
    x = keras.layers.Add()([tensor,x])
    
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =256, kernel_size=(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =256, kernel_size=(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding = 'same')(x)
    
    tensor = Conv_BN(tensor, filters=256, kernel_size = 1,strides=(2,1))
    x = keras.layers.Add()([tensor,x])
    
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =728, kernel_size=(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =728, kernel_size=(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding = 'same')(x)
    
    tensor = Conv_BN(tensor, filters=728, kernel_size = 1,strides=(2,1))
    x = keras.layers.Add()([tensor,x])
    return x

# =============================================================================

def Middle_Flow(tensor):
    
    for _ in range(8):
        x = keras.layers.ReLU()(tensor)
        x = Sep_BN(x, filters = 728, kernel_size = (3,1))
        x = keras.layers.ReLU()(x)
        x = Sep_BN(x, filters = 728, kernel_size = (3,1))
        x = keras.layers.ReLU()(x)
        x = Sep_BN(x, filters = 728, kernel_size = (3,1))
        x = keras.layers.ReLU()(x)
        tensor = keras.layers.Add()([tensor,x])
        
        return tensor

# =============================================================================

def Exit_Flow(tensor):
    
    x = keras.layers.ReLU()(tensor)
    x = Sep_BN(x, filters = 728,  kernel_size=(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters = 1024,  kernel_size=(3,1))
    x = keras.layers.MaxPool2D(pool_size = (3,1), strides = (2,1), padding ='same')(x)
    
    tensor = Conv_BN(tensor, filters =1024, kernel_size=1, strides =(2,1))
    x = keras.layers.Add()([tensor,x])
    
    x = Sep_BN(x, filters = 1536,  kernel_size=(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters = 2048,  kernel_size=(3,1))
    x = keras.layers.GlobalAvgPool2D()(x)
    
    x = keras.layers.Dense (units = 2, activation = 'softmax')(x)
    
    return x
# =============================================================================

def Xception():
    
    input = keras.layers.Input(shape = (6000,1,1))
    
    x = Entry_Flow(input)
    x = Middle_Flow(x)
    output = Exit_Flow(x)
    
    model = Model(inputs=input, outputs=output)
    
    return model

# =============================================================================
def Entry_Flow(x):
    
    x = Conv_BN(x, filters =32, kernel_size =(3,1), strides=(2,1))
    x = keras.layers.ReLU()(x)
    x = Conv_BN(x, filters =64, kernel_size =(3,1), strides=(1,1))
    tensor = keras.layers.ReLU()(x)
    
    x = Sep_BN(tensor, filters = 128, kernel_size =(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters = 128, kernel_size =(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding = 'same')(x)
    
    tensor = Conv_BN(tensor, filters=128, kernel_size = 1,strides=(2,1))
    x = keras.layers.Add()([tensor,x])
    
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =256, kernel_size=(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =256, kernel_size=(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding = 'same')(x)
    
    tensor = Conv_BN(tensor, filters=256, kernel_size = 1,strides=2)
    x = keras.layers.Add()([tensor,x])
    
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =728, kernel_size=(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters =728, kernel_size=(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1), strides=(2,1), padding = 'same')(x)
    
    tensor = Conv_BN(tensor, filters=728, kernel_size = 1,strides=(2,1))
    x = keras.layers.Add()([tensor,x])
    return x


def Middle_Flow_Tiny(tensor):
    
    for _ in range(4):
        x = keras.layers.ReLU()(tensor)
        x = Sep_BN(x, filters = 728, kernel_size = (3,1))
        x = keras.layers.ReLU()(x)
        x = Sep_BN(x, filters = 728, kernel_size = (3,1))
        x = keras.layers.ReLU()(x)
        x = Sep_BN(x, filters = 728, kernel_size = (3,1))
        x = keras.layers.ReLU()(x)
        tensor = keras.layers.Add()([tensor,x])
        
        return tensor
    
# =============================================================================   
    
def Exit_Flow_Tiny(tensor):
    
    x = keras.layers.ReLU()(tensor)
    x = Sep_BN(x, filters = 728,  kernel_size=(3,1))
    x = keras.layers.ReLU()(x)
    x = Sep_BN(x, filters = 512,  kernel_size=(3,1))
    x = keras.layers.MaxPool2D(pool_size = (3,1), strides = (2,1), padding ='same')(x)
    
    tensor = Conv_BN(tensor, filters =512, kernel_size=1, strides =(2,1))
    x = keras.layers.Add()([tensor,x])
    
    x = keras.layers.GlobalAvgPool2D()(x)
    
    x = keras.layers.Dense (units = 2, activation = 'softmax')(x)
    
    return x 
# =============================================================================

def Xception_Tiny():
    
    input = keras.layers.Input(shape = (6000,1,1))
    
    x = Entry_Flow(input)
    x = Middle_Flow_Tiny(x)
    output = Exit_Flow_Tiny(x)
    
    model = Model(inputs=input, outputs=output)
    
    return model

model = Xception_Tiny()   
model.summary()


