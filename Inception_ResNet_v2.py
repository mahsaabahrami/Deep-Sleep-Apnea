import keras

# =============================================================================

def CBN(pl, kn, ks, strides, padding, activation):
    
    '''   
    Parameters
    ----------
    pl : previous layer
    kn : number of kernels 
    ks :kernel size
    strides :  The default is (1,1).
    padding :  The default is 'same'.

    '''
    x = keras.layers.Conv2D(filters=kn, kernel_size = ks, strides=strides, padding=padding, use_bias=False)(pl)
    x = keras.layers.BatchNormalization(axis=3, scale=False)(x)
    if activation: 
       x = keras.layers.Activation(activation='relu')(x)
    
    return x
# =============================================================================

def StemBlock(pl):
    
    x = CBN(pl, kn=32, ks=(3,1), strides=(2,1), padding='valid', activation=True)
    x = CBN(x,  kn=32, ks=(3,1), strides=(1,1), padding='valid', activation=True)
    x = CBN(x,  kn=64, ks=(3,1), strides=(1,1), padding='valid', activation=True)

    x_11 = keras.layers.MaxPooling2D((3,1), strides=(1,1), padding='valid')(x)
    x_12 = CBN(x, kn=64, ks=(3,1), strides=(1,1), padding='valid', activation=True)

    x = keras.layers.Concatenate(axis=3)([x_11,x_12])

    x_21 = CBN(x, kn=64, ks=(1,1), strides=(1,1), padding='same', activation=True)

    x_21 = CBN(x_21, kn=64, ks=(7,1), strides=(1,1), padding='same', activation=True)
    x_21 = CBN(x_21, kn=96, ks=(3,1), strides=(1,1), padding='valid', activation=True)

    x_22 = CBN(x,    kn=64, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x_22 = CBN(x_22, kn=96, ks=(3,1), strides=(1,1), padding='valid', activation=True)

    x = keras.layers.Concatenate(axis=3)([x_21,x_22])

    x_31 = CBN(x, kn=192, ks=(3,1),strides=(1,1), padding='valid', activation=True)
    x_32 = keras.layers.MaxPooling2D((3,1), strides=(1,1), padding='valid')(x)
    x = keras.layers.Concatenate(axis=3)([x_31,x_32])
    
    return x

# =============================================================================    

def Inception_ResNet_A(x, scale):

    x_0 = CBN(x,   kn=32, ks=(1,1), strides=(1,1), padding='same', activation=True)
    
    x_1 = CBN(x,   kn=32, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x_1 = CBN(x_1, kn=32, ks=(3,1), strides=(1,1), padding='same', activation=True)
    
    x_2 = CBN(x,   kn=32, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x_2 = CBN(x_2, kn=48, ks=(3,1), strides=(1,1), padding='same', activation=True)
    x_2 = CBN(x_2, kn=64, ks=(3,1), strides=(1,1), padding='same', activation=True)
    
    x3 = [x_0, x_1, x_2]
    
    x3 = keras.layers.Concatenate(axis=3)(x3)
    
    x3 = CBN(x3, kn=384, ks=(1,1), strides=(1,1), padding='same', activation=False)
    
    x = keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=keras.backend.int_shape(x)[1:],
                      arguments={'scale': scale})([x, x3])
    return x

# =============================================================================  

def Reduction_A(x):
    
    x1 = keras.layers.MaxPooling2D((3,1), strides=(2,1), padding='valid')(x)

    x2 = CBN(x, kn=384, ks=(3,1), strides=(2,1),padding='valid', activation=True)

    x3 = CBN(x,  kn=256, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x3 = CBN(x3, kn=256, ks=(3,1), strides=(1,1), padding='same', activation=True)
    x3 = CBN(x3, kn=384, ks=(3,1), strides=(2,1), padding='valid', activation=True)

    x = keras.layers.Concatenate(axis=3)([x1, x2, x3])
    
    return x

# =============================================================================  
def Inception_ResNet_B(x, scale):
    
    x_0 = CBN(x, kn=192, ks=(1,1), strides=(1,1), padding='same', activation=True)
    
    x_1 = CBN(x,   kn=128, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x_1 = CBN(x_1, kn=192, ks=(7,1), strides=(1,1), padding='same', activation=True)
    
    x3 = [x_0, x_1]
    
    x3 = keras.layers.Concatenate(axis=3)(x3)
    x3 = CBN(x3, kn=1152, ks=(1,1), strides=(1,1), padding='same', activation=False)
    
    x = keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape = keras.backend.int_shape(x)[1:],
                      arguments={'scale': scale})([x, x3])
    return x

# =============================================================================  

def Reduction_B(x):
    
    x1 = keras.layers.MaxPooling2D((3,1), strides=(2,1),padding='valid')(x)

    x2 = CBN(x,  kn=256, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x2 = CBN(x2, kn=384, ks=(3,1), strides=(2,1), padding='valid', activation=True)

    x3 = CBN(x,  kn=256, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x3 = CBN(x3, kn=256, ks=(3,1), strides=(2,1), padding='valid', activation=True)

    x4 = CBN(x,  kn=256, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x4 = CBN(x4, kn=256, ks=(3,1), strides=(1,1), padding='same', activation=True)
    x4 = CBN(x4, kn=256, ks=(3,1), strides=(2,1),padding='valid', activation=True)

    x = keras.layers.Concatenate(axis=3)([x1, x2, x3, x4])
    
    return x

# =============================================================================  

def Inception_ResNet_C(x, scale):

    x_0 = CBN(x, kn=192, ks=(1,1), strides=(1,1), padding='same', activation=True)
    
    x_1 = CBN(x,  kn=192, ks=(1,1), strides=(1,1), padding='same', activation=True)
    x_1 = CBN(x_1,kn=256, ks=(3,1), strides=(1,1), padding='same', activation=True)
    
    x3 = [x_0, x_1]
    
    x3 = keras.layers.Concatenate(axis=3)(x3)
    x3 = CBN(x3, kn=2048, ks=(1,1), strides=(1,1), padding='same', activation=False)
   
    x =  keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                      output_shape=keras.backend.int_shape(x)[1:],
                      arguments={'scale': scale})([x, x3])
    return x

# =============================================================================

def Inception_ResNet_v2():
    input_data = keras.layers.Input(shape=(6000, 1, 1))
    x = StemBlock(input_data)
    
    x = Inception_ResNet_A(x, 0.15)
    x = Inception_ResNet_A(x, 0.15)
    
    x = Reduction_A(x)
    
    x = Inception_ResNet_B(x, 0.1)
    x = Inception_ResNet_B(x, 0.1)
    
    x = Reduction_B(x)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = keras.layers.Dropout(0.6)(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(input_data, x)
    
    return model

#==============================================================================

def Inception_ResNet_v2_Tiny():
    input_data = keras.layers.Input(shape=(6000, 1, 1))
    x = StemBlock(input_data)
    
    x = Inception_ResNet_A(x, 0.15)
    x = Inception_ResNet_A(x, 0.15)
    
    x = Reduction_A(x)
    
    #x = Inception_ResNet_B(x, 0.1)
    #x = Inception_ResNet_B(x, 0.1)
    
    x = Reduction_B(x)
    
    x = keras.layers.GlobalAveragePooling2D()(x)
    
    x = keras.layers.Dropout(0.6)(x)
    x = keras.layers.Dense(2, activation='softmax')(x)
    
    model = keras.Model(input_data, x)
    
    return model

#==============================================================================

model=Inception_ResNet_v2_Tiny()
model.summary()
    