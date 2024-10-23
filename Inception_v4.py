# =============================================================================
# import libraries:
from keras.models import Model
import keras   
# =============================================================================

def CBN(pl , kn , ks , strides = (1,1) , padding = 'valid'):
    
    '''
    Parameters
    ----------
    pl : previous layer
    kn :number of kernels
    ks : kernel size
    strides : The default is (1,1).
    padding : The default is 'valid'.

    '''
    x = keras.layers.Conv2D(kn, kernel_size= ks, 
                            strides=strides , padding=padding) (pl)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation = 'relu') (x)
    
    return x

# =============================================================================

def stemBlock(pl):
    
    x = CBN(pl, kn = 32, ks = (3,1), strides = (2,1))
    x = CBN(x,  kn = 32, ks = (3,1))
    x = CBN(x,  kn = 64, ks = (3,1))
    
    x_1 = CBN(x, kn = 96, ks = (3,1), strides = (2,1))
    x_2 = keras.layers.MaxPool2D(pool_size=(3,1) , strides=(2,1)) (x)
    
    x = keras.layers.concatenate([x_1 , x_2], axis = 3)
    
    x_1 = CBN(x,   kn = 64, ks = (1,1))
    x_1 = CBN(x_1, kn = 64, ks = (7,1), padding ='same')
    x_1 = CBN(x_1, kn = 96, ks = (3,1))
    
    x_2 = CBN(x,   kn = 64, ks = (1,1))
    x_2 = CBN(x_2, kn = 96, ks = (3,1))
    
    x = keras.layers.concatenate([x_1 , x_2], axis = 3)
    
    x_1 = CBN(x, kn = 192, ks = (3,1) , strides=2)
    x_2 = keras.layers.MaxPool2D(pool_size=(3,1) , strides=(2,1) ) (x)
    
    x = keras.layers.concatenate([x_1 , x_2], axis = 3)
    
    return x

# =============================================================================

def reduction_A_Block(pl) :
    
    x_1 = CBN(pl = pl,  kn = 192, ks = (1,1))
    x_1 = CBN(pl = x_1, kn = 224, ks = (3,1) , padding='same')
    x_1 = CBN(pl = x_1, kn = 256, ks = (3,1) , strides=(2,1)) 
    
    x_2 = CBN(pl = pl, kn = 384, ks = (3,1) , strides=(2,1))
    
    x_3 = keras.layers.MaxPool2D(pool_size=(3,1) , strides=(2,1)) (pl)
    
    x = keras.layers.concatenate([x_1 , x_2 , x_3], axis = 3)
    
    return x

# =============================================================================

def reduction_B_Block(pl):
    
    x_1 = keras.layers.MaxPool2D(pool_size=(3,1) , strides=(2,1))(pl)
    
    x_2 = CBN(pl = pl,  kn = 192, ks = (1,1))
    x_2 = CBN(pl = x_2, kn = 192, ks = (3,1) , strides=(2,1))
    
    x_3 = CBN(pl = pl, kn = 256, ks = (1,1))

    x_3 = CBN(pl = x_3, kn = 320, ks = (7,1) , padding='same')
    x_3 = CBN(pl = x_3, kn = 320, ks = (3,1) , strides=(2,1))
    
    x = keras.layers.concatenate([x_1 , x_2 , x_3], axis = 3)
    
    return x

# =============================================================================

def InceptionBlock_A(pl): 
    
    x_1 = CBN(pl = pl,  kn = 64, ks = (1,1))
    x_1 = CBN(pl = x_1, kn = 96, ks = (3,1) , strides=(1,1), padding='same' )
    x_1 = CBN(pl = x_1, kn = 96, ks = (3,1) , strides=(1,1) , padding='same')
    
    x_2 = CBN(pl = pl,  kn = 64, ks = (1,1))
    x_2 = CBN(pl = x_2, kn = 96, ks = (3,1) , padding='same')
    
    x_3 = keras.layers.AveragePooling2D(pool_size=(3,1) , strides=1 , padding='same')(pl)
    x_3 = CBN(pl = x_3, kn = 96, ks = (1,1) , padding='same')
    
    x_4 = CBN(pl = pl, kn = 96, ks = (1,1))
    
    output = keras.layers.concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)

    return output

# =============================================================================
   
def InceptionBlock_B(pl):
    
    x_1 = CBN(pl = pl,  kn = 192, ks = (1,1))
    x_1 = CBN(pl = x_1, kn = 224, ks = (7,1), padding='same')
   
    x_1 = CBN(pl = x_1, kn = 256, ks = (7,1), padding='same')
    
    
    x_2 = CBN(pl = pl,  kn = 192, ks = (1,1))
    x_2 = CBN(pl = x_2, kn = 224, ks = (7,1), padding='same')
    x_2 = CBN(pl = x_2, kn = 256, ks = (7,1), padding='same')
    
    x_3 = keras.layers.AveragePooling2D(pool_size=(3,1), strides=1, padding='same')(pl)
    x_3 = CBN(pl = x_3, kn = 128, ks = (1,1))
    
    x_4 = CBN(pl = pl, kn = 384, ks = (1,1))

    output = keras.layers.concatenate([x_1 , x_2 ,x_3, x_4], axis = 3) 
    
    return output

# =============================================================================

def InceptionBlock_C(pl):
    
    x_1 =   CBN(pl = pl,  kn = 384, ks = (1,1))
    x_1 =   CBN(pl = x_1, kn = 512, ks = (3,1) , padding='same')
    x_1_1 = CBN(pl = x_1, kn = 256, ks = (3,1), padding='same')
    x_1_2 = CBN(pl = x_1, kn = 256, ks = (3,1), padding='same')
    x_1 = keras.layers.concatenate([x_1_1 , x_1_2], axis = 3)
    
    x_2 =   CBN(pl = pl,  kn = 384, ks = (1,1))
    x_2_1 = CBN(pl = x_2, kn = 256, ks = (3,1), padding='same')
    x_2_2 = CBN(pl = x_2, kn = 256, ks = (3,1), padding='same')
    x_2 = keras.layers.concatenate([x_2_1 , x_2_2], axis = 3)
    
    x_3 = keras.layers.MaxPool2D(pool_size=(3,1),strides = 1 , padding='same')(pl)
    x_3 = CBN(pl = x_3, kn = 256, ks = (3,1) , padding='same')
    
    x_4 = CBN(pl = pl, kn = 256, ks = (1,1))
    
    output = keras.layers.concatenate([x_1 , x_2 , x_3 , x_4], axis = 3)
    
    return output

# =============================================================================

def Inceptionv4():
    
    input_layer = keras.layers.Input(shape=(6000 , 1 , 1))
    
    x = stemBlock(pl=input_layer)
    
    x = InceptionBlock_A(pl=x)
    x = InceptionBlock_A(pl=x)
    x = InceptionBlock_A(pl=x)
    x = InceptionBlock_A(pl=x)
    
    x = reduction_A_Block(pl=x)
    
    x = InceptionBlock_B(pl=x)
    x = InceptionBlock_B(pl=x)
    x = InceptionBlock_B(pl=x)

    
    x = reduction_B_Block(pl= x)
    
    x = InceptionBlock_C(pl=x)
    x = InceptionBlock_C(pl=x)

    
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = keras.layers.Dropout(rate = 0.8) (x)
    x = keras.layers.Dense(units = 2, activation='softmax')(x)
    
    model = Model(inputs = input_layer , outputs = x , name ='Inception-V4')
    
    return model

# =============================================================================

def Inceptionv4_Tiny():
    
    input_layer = keras.layers.Input(shape=(6000 , 1 , 1))
    
    x = stemBlock(pl=input_layer)
    
    x = InceptionBlock_A(pl=x)
    x = InceptionBlock_A(pl=x)
    x = InceptionBlock_A(pl=x)
    #x = InceptionBlock_A(pl=x)
    
    x = reduction_A_Block(pl=x)
    
    x = InceptionBlock_B(pl=x)
    x = InceptionBlock_B(pl=x)
    #x = InceptionBlock_B(pl=x)

    
    x = reduction_B_Block(pl= x)
    
    x = InceptionBlock_C(pl=x)
   # x = InceptionBlock_C(pl=x)

    
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = keras.layers.Dropout(rate = 0.8) (x)
    x = keras.layers.Dense(units = 2, activation='softmax')(x)
    
    model = Model(inputs = input_layer , outputs = x , name ='Inception-V4')
    
    return model

# =============================================================================

model = Inceptionv4_Tiny()
model.summary()