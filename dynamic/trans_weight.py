from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import random
import numpy as np
import warnings
from sklearn.metrics import precision_recall_fscore_support,roc_auc_score

batch_size = 128
num_classes = 2
epochs =5


#training data
X = np.genfromtxt('./embedding_scale5k_end/embedding_17.csv', delimiter=',', dtype=float, autostrip=True)

if X.shape[1] == 129:
    warnings.warn('Output contains {} columns, {} expected!'
                  '\nAssuming first column to be the nodeID...'
                  .format(X.shape[1],   128), Warning)
    # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
    keys = map(str, np.array(X[:, 0], dtype=int))
    X = dict(zip(keys, X[:, 1:]))


#load_data=open('./data_scale10k/data0_weight.txt','r')
load_data = open('./delta_data_weight_scale10k_end/delta_data8_weight_end.txt','r')
raw_data=load_data.read()
tmp = raw_data.splitlines()

x_train=np.zeros((19000,256))
y_train=np.zeros((19000,1))

list1=[]
list2=[]

for i,row in enumerate(tmp):
    pair = np.fromstring(row, dtype=int, sep=',')
    #print (X[str(pair[0])])
    try:
        x_train[i] = np.concatenate((X[str(pair[0])], X[str(pair[1])]),axis=0)
        if pair[2] > 10:
            y_train[i] = 0
        # elif pair[2] > 50:
        #     y_train[i] = 0.5
        else:
            y_train[i] = 1
        if pair[0] not in list1:
            list1.append(pair[0])
        if pair[1] not in list1:
            list1.append(pair[1])
        list2.append((pair[0],pair[1]))
    except:
        continue


while i<19000:
    a=random.sample(list1,2)
    try:
        if a not in list2:
            i = i + 1
            x_train[i] = np.concatenate((X[str(a[0])], X[str(a[1])]),axis=0)
            y_train[i] = 0

    except:
        continue



#test data
X1 = np.genfromtxt('./embedding_scale5k_end/embedding_19.csv', delimiter=',', dtype=float, autostrip=True)


if X1.shape[1] == 129:
    warnings.warn('Output contains {} columns, {} expected!'
                  '\nAssuming first column to be the nodeID...'
                  .format(X1.shape[1], 128), Warning)
    # Assume first col is node id and rest are embedding features [id, X_0, X_1, ..., X_D]
    keys = map(str, np.array(X1[:, 0], dtype=int))
    X1 = dict(zip(keys, X1[:, 1:]))

load_data1 = open('./delta_data_weight_scale10k_end/delta_data9_weight_end.txt', 'r')
raw_data1 = load_data1.read()
tmp1 = raw_data1.splitlines()

x_test = np.zeros((19000, 256))
y_test = np.zeros((19000, 1))

list3 = []
list4 = []

for j, row in enumerate(tmp1):
    pair = np.fromstring(row, dtype=int, sep=',')
    # print (X[str(pair[0])])
    try:
        x_test[j] = np.concatenate((X1[str(pair[0])], X1[str(pair[1])]), axis=0)
        if pair[2] > 10:
            y_test[j] = 0
        #elif pair[2] > 100:
        #    y_test[j] = 0.5
        else:
            y_test[j] = 1
        if pair[0] not in list3:
            list3.append(pair[0])
        if pair[1] not in list3:
            list3.append(pair[1])
        list4.append((pair[0], pair[1]))
    except:
        continue

while j < 19000:
    b = random.sample(list3, 2)
    try:
        if b not in list4:
            j = j + 1
            x_test[j] = np.concatenate((X1[str(b[0])], X1[str(b[1])]), axis=0)
            y_test[j] = 0

    except:
        continue



x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()


# model.add(Conv2D(32, kernel_size=(3, 3),
#                  activation='relu',
#                  input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(num_classes, activation='softmax'))


#model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


arg_test=np.argmax(y_test,axis=1)
arg_predict=np.argmax(model.predict(x_test),axis=1)

arg_test=np.reshape(arg_test,19000)
arg_predict=np.reshape(arg_predict,19000)


# print (np.array(np.where(arg_test==0)).shape)
# print (np.array(np.where(arg_test==1)).shape)
# print (np.array(np.where(arg_test==2)).shape)
#
# print (np.array(np.where(arg_predict==0)).shape)
# print (np.array(np.where(arg_predict==1)).shape)
# print (np.array(np.where(arg_predict==2)).shape)



precision, recall, f1score,_=precision_recall_fscore_support(arg_test,arg_predict,average='macro')

print (precision,recall,f1score)

aucroc=roc_auc_score(arg_test,arg_predict,average='macro')

print (aucroc)