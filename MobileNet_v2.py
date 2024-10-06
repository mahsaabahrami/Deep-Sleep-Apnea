# =============================================================================
# Import Libraries
import keras
from tensorflow.keras import Model
# =============================================================================
def Expansion_Block(x,t,filters,block_id):
    
    total_filters = t*filters
    x = keras.layers.Conv2D(total_filters,1, padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    
    return x
# =============================================================================
def Depthwise_Block(x,stride,block_id):

    x = keras.layers.DepthwiseConv2D(3, strides=stride, padding ='same', use_bias = False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    
    return x
# =============================================================================
def Projection_Block(x,out_channels,block_id):

    x = keras.layers.Conv2D(filters=out_channels,kernel_size = 1,  padding='same',use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    
    return x

# =============================================================================
def Bottleneck(x,t,filters, out_channels,stride,block_id):

    y = Expansion_Block(x,t,filters,block_id)
    y = Depthwise_Block(y,stride,block_id)
    y = Projection_Block(y, out_channels,block_id)
    
    if y.shape[-1]==x.shape[-1]:
       y = keras.layers.add([x,y])
    
    return y 
# =============================================================================

def MobileNet_v2(input_shape = (6000,1,1), n_classes=2):
    
    input = keras.layers.Input (input_shape)
    
    x = keras.layers.Conv2D(32,3,strides = 2,padding='same', use_bias=False)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 16, stride = 1,block_id = 1)
  
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)   
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)

    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)
 
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)   
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)
    
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)   
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)
    
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)  
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)
    
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 320, stride = 1,block_id = 17)
    
    x = keras.layers.Conv2D(filters = 1280,kernel_size = 1,padding='same',use_bias=False, name = 'last_conv')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
 
    
    output = keras.layers.Dense(n_classes,activation='softmax')(x)
    
    model = Model(input, output)
    
    return model
# =============================================================================

def MobileNet_v2_Tiny(input_shape = (6000,1,1), n_classes=2):
    
    input = keras.layers.Input (input_shape)
    
    x = keras.layers.Conv2D(32,3,strides = 2,padding='same', use_bias=False)(input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 16, stride = 1,block_id = 1)
  
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)   
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)

    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)
 
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)   
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)
    
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)   
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)
    
    
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)  
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)

    
    x = keras.layers.Conv2D(filters = 320,kernel_size = 1,padding='same',use_bias=False, name = 'last_conv')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    
    output = keras.layers.Dense(n_classes,activation='softmax')(x)
    
    model = Model(input, output)
    
    return model
# =============================================================================
model=MobileNet_v2_Tiny(input_shape = (6000,1,1), n_classes=2)
model.summary()