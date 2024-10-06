# =============================================================================
# Import libraries
import keras
from tensorflow.keras import Model
# =============================================================================

def Building_Block (x, filters, strides):
    
    x = keras.layers.DepthwiseConv2D(kernel_size = (3,1), strides = strides, padding = 'same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    
    x = keras.layers.Conv2D(filters = filters, kernel_size = (1,1), strides = (1,1))(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    
    return x

# =============================================================================
 
def MobileNet_v1():
   input = keras.layers.Input(shape = (6000,1,1))
  
   x = keras.layers.Conv2D(filters = 32, kernel_size = (3,1), strides = (2,1), padding = 'same')(input)
   x = keras.layers.BatchNormalization()(x)
   x = keras.layers.ReLU()(x)

   x = Building_Block(x, filters = 64, strides = 1)   
   x = Building_Block(x, filters = 128, strides = 2)   
   x = Building_Block(x, filters = 128, strides = 1) 
   x = Building_Block(x, filters = 256, strides = 2)
   x = Building_Block(x, filters = 256, strides = 1) 
   x = Building_Block(x, filters = 512, strides = 2)  
   for _ in range (5):
      x = Building_Block(x, filters = 512, strides = 1) 
   x = Building_Block(x, filters = 512, strides = 1) 
   x = Building_Block(x, filters = 1024, strides = 1)  
   x = keras.layers.Flatten()(x)

   output = keras.layers.Dense (units = 2, activation = 'softmax')(x)
   
   model = Model(inputs=input, outputs=output)
   
   return model  
# =============================================================================

def MobileNet_v1_Tiny():

   input = keras.layers.Input(shape = (6000,1,1))
  
   x = keras.layers.Conv2D(filters = 32, kernel_size = (3,1), strides = (2,1), padding = 'same')(input)
   x = keras.layers.BatchNormalization()(x)
   x = keras.layers.ReLU()(x)

   x = Building_Block(x, filters = 64, strides = 1)   
   x = Building_Block(x, filters = 128, strides = 2)   
   x = Building_Block(x, filters = 128, strides = 1) 
   x = Building_Block(x, filters = 256, strides = 2)
   x = Building_Block(x, filters = 256, strides = 1) 
   x = Building_Block(x, filters = 512, strides = 2)  
   for _ in range (3):
      x = Building_Block(x, filters = 512, strides = 1) 

   x = keras.layers.Flatten()(x)

   output = keras.layers.Dense (units = 2, activation = 'softmax')(x)
   
   model = Model(inputs=input, outputs=output)
   
   return model  

model=MobileNet_v1_Tiny()
model.summary()
