# =============================================================================
import keras
from tensorflow.keras import Model
# =============================================================================
def Identity_Block(X, f, filters, stage, block):
        
    F1, F2, F3 = filters
    
    X_shortcut = X
        
    X = keras.layers.Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = keras.layers.BatchNormalization(axis = 3)(X)
    X = keras.layers.ReLU()(X)
        
    X = keras.layers.Conv2D(filters = F2, kernel_size = (f, 1), strides = (1,1), padding = 'same')(X)
    X = keras.layers.BatchNormalization(axis = 3)(X)
    X = keras.layers.ReLU()(X)

    X = keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = keras.layers.BatchNormalization(axis = 3)(X)


    X = keras.layers.Add()([X_shortcut, X])
    X = keras.layers.ReLU()(X)
        
    return X
# =============================================================================
def Convolutional_Block(X, f, filters, stage, block, s = 2):
        

    F1, F2, F3 = filters
    
    X_shortcut = X
    
    X = keras.layers.Conv2D(F1, (1, 1), strides = (s,1))(X)
    X = keras.layers. BatchNormalization(axis = 3)(X)
    X = keras.layers.ReLU()(X)
    X = keras.layers.Conv2D(filters = F2, kernel_size = (f, 1), strides = (1,1), padding = 'same')(X)
    X = keras.layers.BatchNormalization(axis = 3)(X)
    X = keras.layers.ReLU()(X)
    X = keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid')(X)
    X = keras.layers.BatchNormalization(axis = 3)(X)
    
    X_shortcut = keras.layers.Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,1), padding = 'valid')(X_shortcut)
    X_shortcut = keras.layers.BatchNormalization(axis = 3)(X_shortcut)
    X = keras.layers.Add()([X_shortcut, X])
    X = keras.layers.ReLU()(X)
   
    return X
# =============================================================================
def ResNet50(input_shape = (6000, 1, 1)):
    
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Conv2D(64, (7, 1), strides = (2, 1), name = 'conv1')(X_input)
    X = keras.layers.BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = keras.layers.ReLU()(X)
    X = keras.layers.MaxPooling2D((3, 1), strides=(2, 1))(X)
    
    
    X = Convolutional_Block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 2)
    X = Identity_Block(X, 3, [64, 64, 256], stage=2, block='b')
    X = Identity_Block(X, 3, [64, 64, 256], stage=2, block='c')


    X = Convolutional_Block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = Identity_Block(X, 3, [128, 128, 512], stage=3, block='b')
    X = Identity_Block(X, 3, [128, 128, 512], stage=3, block='c')
    X = Identity_Block(X, 3, [128, 128, 512], stage=3, block='d')


    X = Convolutional_Block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    
    X = Convolutional_Block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = keras.layers.AveragePooling2D(pool_size=(2, 1))(X)
    
    
    X = keras.layers.Flatten()(X)
    
    X = keras.layers.Dense(2, activation='softmax')(X)
    
    model = Model(inputs = X_input, outputs = X)
    return model

def ResNet50_Tiny(input_shape = (6000, 1, 1)):
    
    X_input = keras.layers.Input(input_shape)

    X = keras.layers.Conv2D(64, (7, 1), strides = (2, 1))(X_input)
    X = keras.layers.BatchNormalization(axis = 3)(X)
    X = keras.layers.ReLU()(X)
    X = keras.layers.MaxPooling2D((3, 1), strides=(2, 1))(X)
    X = Convolutional_Block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 2)
    X = Identity_Block(X, 3, [64, 64, 256], stage=2, block='b')
    X = Identity_Block(X, 3, [64, 64, 256], stage=2, block='c')

    X = Convolutional_Block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = Identity_Block(X, 3, [128, 128, 512], stage=3, block='b')
    X = Identity_Block(X, 3, [128, 128, 512], stage=3, block='c')
    X = Identity_Block(X, 3, [128, 128, 512], stage=3, block='d')

    X = Convolutional_Block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = Identity_Block(X, 3, [256, 256, 1024], stage=4, block='f')
    
    X = keras.layers.AveragePooling2D(pool_size=(2, 1),name='avg_pool')(X)
    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(2, activation='softmax')(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')
    return model

model=ResNet50_Tiny(input_shape = (6000, 1, 1))
model.summary()