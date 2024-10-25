##-----------------------------------------------------------------------------    
import pickle
import numpy as np
import os
from keras.callbacks import LearningRateScheduler,EarlyStopping
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from DenseNet_121 import DenseNet_121
from DenseNet_169 import DenseNet_169
# =============================================================================

scaler = lambda arr: (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

# =============================================================================

def load_data():
    with open(os.path.join("apnea-ecg.pkl"), 'rb') as f:
        apnea_ecg = pickle.load(f)

    x_train = []
    o_train, y_train = apnea_ecg["x"], apnea_ecg["y"]
    for i in range(len(o_train)):
        x_train1 = o_train[i]
        x_train.append([x_train1])
    x_train = np.array(x_train, dtype="float32")
    x_final=np.array(x_train, dtype="float32").transpose((0,3,1,2))  
    return x_final, y_train

# =============================================================================
def lr_schedule(epoch, lr):
    if epoch > 30 and \
            (epoch - 1) % 10 == 0:
        lr *= 0.1
    print("Learning rate: ", lr)
    return lr

# =============================================================================

# Compile and evaluate model: 
if __name__ == "__main__":
    X, Y = load_data()
    Y = tf.keras.utils.to_categorical(Y, num_classes=2)
    kfold = StratifiedKFold(n_splits=5, shuffle=True,random_state=7)
    cvscores = []
    ACC=[]
    SN=[]
    SP=[]
    F2=[]

    for train, test in kfold.split(X, Y.argmax(1)):
     model1 = DenseNet_121()
     #model2 = DenseNet_169()
     M=[model1]
     #, model2] 
     for model in M:

      model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
      lr_scheduler = LearningRateScheduler(lr_schedule)
      callback1 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
      X1,x_val,Y1,y_val=train_test_split(X[train],Y[train],test_size=0.10,shuffle=True)
      model.fit(X1, Y1, batch_size=64, epochs=100, validation_data=(x_val, y_val),
                        callbacks=[callback1,lr_scheduler])  

      loss, accuracy = model.evaluate(X[test], Y[test]) # test the model
      
      print("Test loss: ", loss)
      print("Accuracy: ", accuracy)
      
      y_score = model.predict(X[test])
      y_predict= np.argmax(y_score, axis=-1)
      
      y_original = np.argmax(Y[test], axis=-1)
     
      C = confusion_matrix(y_original, y_predict, labels=(1, 0))
      
      TP, TN, FP, FN = C[0, 0], C[1, 1], C[1, 0], C[0, 1]
      acc, sn, sp = 1. * (TP + TN) / (TP + TN + FP + FN), 1. * TP / (TP + FN), 1. * TN / (TN + FP)
      f2=f1_score(y_original, y_predict)
      
      print("acc: {}, sn: {}, sp: {}, F:{}".format(acc, sn, sp,f2))
      ACC.append(acc * 100)
      SN.append(sn * 100)
      SP.append(sp * 100)
      F2.append(f2 * 100)
    
    print("acc: {}, sn: {}, sp: {}, F:{}".format(acc, sn, sp,f2))
print("acc: {}, sn: {}, sp: {}, F:{}".format(np.mean(ACC), np.mean(SN),np.mean(SP),np.mean(F2)))
