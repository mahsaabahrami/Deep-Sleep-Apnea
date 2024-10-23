# =============================================================================
# Import Libraries:
import keras
from keras.models import Model
# =============================================================================

def CBN(pl , kn , ks , strides =(1,1) , padding = 'same'):
    
    '''   
    Parameters
    ----------
    pl : previous layer
    kn : number of kernels 
    ks :kernel size
    strides :  The default is (1,1).
    padding :  The default is 'same'.

    '''
    x = keras.layers.Conv2D(filters=kn, kernel_size = ks, strides=strides , padding=padding)(pl)
    x = keras.layers.BatchNormalization(axis=3)(x)
    x = keras.layers.Activation(activation='relu')(x)
    
    return x

# =============================================================================

def StemBlock(pl):
    
    x = CBN(pl, kn = 32, ks=(3,1) , strides=(2,1))
    x = CBN(x, kn = 32, ks=(3,1))
    x = CBN(x, kn = 64, ks=(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1) , strides=(2,1)) (x)
    
    x = CBN(x, kn = 80, ks=(1,1))
    x = CBN(x, kn = 192, ks=(3,1))
    x = keras.layers.MaxPool2D(pool_size=(3,1) , strides=(2,1)) (x) 
    
    return x    

# =============================================================================    

def InceptionBlock_A(pl,kn):
    
    branch1 = CBN(pl, kn = 64, ks = (1,1))
    branch1 = CBN(branch1, kn=96, ks=(3,1))
    branch1 = CBN(branch1, kn=96, ks=(3,1))
    
    branch2 = CBN(pl, kn=48, ks=(1,1))
    branch2 = CBN(branch2, kn=64, ks=(3,1)) 
    
    branch3 = keras.layers.AveragePooling2D(pool_size=(3,1) , strides=(1,1) , padding='same') (pl)
    branch3 = CBN(branch3, kn = kn, ks = (1,1))
    
    branch4 = CBN(pl, kn=64, ks=(1,1)) 
    output = keras.layers.concatenate([branch1 , branch2 , branch3 , branch4], axis=3)
    return output

# =============================================================================

def InceptionBlock_B(pl,kn):
    
    branch1 = CBN(pl, kn = kn, ks = (1,1))
    branch1 = CBN(branch1, kn = kn, ks = (7,1))
    branch1 = CBN(branch1, kn = kn, ks = (7,1))    
    
    branch2 = CBN(pl, kn = kn, ks = (1,1))
    branch2 = CBN(branch2, kn = 192, ks = (7,1))
    
    branch3 = keras.layers.AveragePooling2D(pool_size=(3,1) , strides=(1,1) , padding ='same') (pl)
    branch3 = CBN(branch3, kn = 192, ks = (1,1))
    
    branch4 = CBN(pl, kn = 192, ks = (1,1))
    
    output = keras.layers.concatenate([branch1 , branch2 , branch3 , branch4], axis = 3)
    
    return output    

# =============================================================================
    
def ReductionBlock_A(pl):
    
    branch1 = CBN(pl, kn = 64, ks = (1,1))
    branch1 = CBN(branch1, kn = 96, ks = (3,1))
    branch1 = CBN(branch1, kn = 96, ks = (3,1) , strides=(2,1) ) #, padding='valid'
    
    branch2 = CBN(pl, kn = 384, ks=(3,1) , strides=(2,1) )
    
    branch3 = keras.layers.MaxPool2D(pool_size=(3,1) , strides=(2,1) , padding='same')(pl)
    
    output = keras.layers.concatenate([branch1 , branch2 , branch3], axis = 3)
    
    return output
  
# =============================================================================


def auxiliary_classifier(pl):
    
    x = keras.layers.AveragePooling2D(pool_size=(5,1) , strides=(3,1)) (pl)
    x = CBN(x, kn = 128, ks = (1,1))
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(rate = 0.2) (x)
    x = keras.layers.Dense(units = 2, activation='softmax') (x)
    
    return x

# =============================================================================

def InceptionV3():
    
    input_layer = keras.layers.Input(shape=(6000 , 1 , 1))
    
    x = StemBlock(input_layer)
    
    x = InceptionBlock_A(pl = x ,kn = 32)
    x = InceptionBlock_A(pl = x ,kn = 64)
    x = InceptionBlock_A(pl = x ,kn = 64)
    
    x = ReductionBlock_A(pl = x )
    
    x = InceptionBlock_B(pl = x  , kn = 128)
    x = InceptionBlock_B(pl = x , kn = 160)
    x = InceptionBlock_B(pl = x , kn = 160)
    x = InceptionBlock_B(pl = x , kn = 192)
    
    Aux = auxiliary_classifier(pl = x)
    
    model = Model(inputs = input_layer , outputs = [Aux] , name = 'Inception-V3')
    
    return model
# =============================================================================

def InceptionV3_Tiny():
    
    input_layer = keras.layers.Input(shape=(6000 , 1 , 1))
    
    x = StemBlock(input_layer)
    
    x = InceptionBlock_A(pl = x ,kn = 32)
    x = InceptionBlock_A(pl = x ,kn = 64)
    x = InceptionBlock_A(pl = x ,kn = 64)
    
    x = ReductionBlock_A(pl = x )
    
    x = InceptionBlock_B(pl = x  , kn = 128)
    x = InceptionBlock_B(pl = x , kn = 160)
    #x = InceptionBlock_B(pl = x , kn = 160)
    #x = InceptionBlock_B(pl = x , kn = 192)
    
    Aux = auxiliary_classifier(pl = x)
    
    model = Model(inputs = input_layer , outputs = [Aux] , name = 'Inception-V3_Tiny')
    
    return model
# =============================================================================


