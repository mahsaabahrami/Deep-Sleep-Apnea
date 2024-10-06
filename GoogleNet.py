# =============================================================================
# import libraries:
import keras
from tensorflow.keras import Model
from keras.initializers import glorot_uniform
# =============================================================================

def Inception_Block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 

  path1 = keras.layers.Conv2D(filters=f1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)

  path2 = keras.layers.Conv2D(filters = f2_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path2 = keras.layers.Conv2D(filters = f2_conv3, kernel_size = (3,1), padding = 'same', activation = 'relu')(path2)

  path3 = keras.layers.Conv2D(filters = f3_conv1, kernel_size = (1,1), padding = 'same', activation = 'relu')(input_layer)
  path3 = keras.layers.Conv2D(filters = f3_conv5, kernel_size = (5,1), padding = 'same', activation = 'relu')(path3)

  path4 = keras.layers.MaxPooling2D((3,1), strides= (1,1), padding = 'same')(input_layer)
  path4 = keras.layers.Conv2D(filters = f4, kernel_size = (1,1), padding = 'same', activation = 'relu')(path4)

  output_layer = keras.layers.concatenate([path1, path2, path3, path4], axis = -1)

  return output_layer
# =============================================================================
def GoogLeNet():

  input_layer = keras.layers.Input(shape = (6000,1, 1))
  
  X =  keras.layers.Conv2D(filters = 64, kernel_size = (7,1), strides = (2,1), padding = 'same', activation = 'relu')(input_layer)
  X =  keras.layers.MaxPooling2D(pool_size = (3,1), strides = (2,1),padding = 'same')(X)
  
  X =  keras.layers.Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
  X =  keras.layers.Conv2D(filters = 192, kernel_size = (3,1),strides = 1, padding = 'same', activation = 'relu')(X)
  X =  keras.layers.MaxPooling2D(pool_size= (3,1), strides = (2,1),padding = 'same')(X)
  
  X = Inception_Block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
  X = Inception_Block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
  
  X = keras.layers.MaxPooling2D(pool_size= (3,1), strides = (2,1),padding='same')(X)
  
  X = Inception_Block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)

  X = Inception_Block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
  X = Inception_Block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
  X = Inception_Block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)
  X = Inception_Block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
  
  X = keras.layers.MaxPooling2D(pool_size = (3,1), strides = (2,1))(X)
  X = Inception_Block(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
  X = Inception_Block(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)
 
  X = keras.layers.GlobalAveragePooling2D(name = 'GAPL')(X)
  
  X = keras.layers.Dropout(0.4)(X)
  X = keras.layers.Dense(2, activation = 'softmax')(X)
  
  model = Model(input_layer, X, name = 'GoogLeNet')

  return model
# =============================================================================

def GoogLeNet_Tiny_v1():

  input_layer = keras.layers.Input(shape = (6000,1, 1))
  
  X = keras.layers.Conv2D(filters = 64, kernel_size = (7,1), strides = (2,1), padding = 'same', activation = 'relu')(input_layer)
  X = keras.layers.MaxPooling2D(pool_size = (3,1), strides = (2,1),padding = 'same')(X)
 
  
  X = keras.layers.Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
  X = keras.layers.Conv2D(filters = 192, kernel_size = (3,1),strides = 1, padding = 'same', activation = 'relu')(X)
  X = keras.layers.MaxPooling2D(pool_size= (3,1), strides = (2,1),padding = 'same')(X)
  
  
  X = Inception_Block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
  X = Inception_Block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
  X = keras.layers.MaxPooling2D(pool_size= (3,1), strides = (2,1),padding='same')(X)
  X = Inception_Block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)


  X = Inception_Block(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
  X = Inception_Block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
  X = Inception_Block(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)
  X = keras.layers.AveragePooling2D(pool_size = (5,1), strides = (3,1))(X)
  
  X = keras.layers.Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X)
  X = keras.layers.Flatten()(X)

  X = keras.layers.Dropout(0.7)(X)
  X = keras.layers.Dense(2, activation = 'softmax')(X)
  
  model = Model(input_layer, X, name = 'Tiny_GoogLeNet')

  return model


# =============================================================================
def GoogLeNet_Tiny_v2():
 
  input_layer = keras.layers.Input(shape = (6000,1, 1))
  
  X = keras.layers.Conv2D(filters = 64, kernel_size = (7,1), strides = (2,1), padding = 'same', activation = 'relu')(input_layer)
  X = keras.layers.MaxPooling2D(pool_size = (3,1), strides = (2,1),padding = 'same')(X)

  
  X = keras.layers.Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
  X = keras.layers.Conv2D(filters = 192, kernel_size = (3,1),strides = 1, padding = 'same', activation = 'relu')(X)
  X =keras.layers.MaxPooling2D(pool_size= (3,1), strides = (2,1),padding = 'same')(X)

  
  X = Inception_Block(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
  X = Inception_Block(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)
  X = keras.layers.MaxPooling2D(pool_size= (3,1), strides = (2,1),padding='same')(X)
  
  
  X = Inception_Block(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)
  X = keras.layers.AveragePooling2D(pool_size = (5,1), strides = (3,1))(X)
  
  X = keras.layers.Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X)
  X = keras.layers.Flatten()(X)
  X = keras.layers.Dropout(0.7)(X)
  X = keras.layers.Dense(2, activation = 'softmax')(X)
  model = Model(input_layer, X, name = 'GoogLeNet')

  return model
model= GoogLeNet_Tiny_v2()
model.summary()