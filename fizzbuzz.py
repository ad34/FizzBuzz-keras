from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras.optimizers import RMSprop
from keras.utils import np_utils
import numpy
from keras.callbacks import Callback,EarlyStopping


num_digits = 10
nb_classes = 4
batch_size = 128

def fb_encode(i):
    if   i % 15 == 0: return [3]
    elif i % 5  == 0: return [2]
    elif i % 3  == 0: return [1]
    else:             return [0]

def bin_encode(i):
    return [i >> d & 1 for d in range(num_digits)]

def fizz_buzz(i, pred):
    return [str(i), "fizz", "buzz", "fizzbuzz"][pred.argmax()]

def create_dataset():
    dataX,dataY = [],[]
    for i in range(101,1024):
         dataX.append(bin_encode(i))
         dataY.append(fb_encode(i))

    return numpy.array(dataX),np_utils.to_categorical(numpy.array(dataY), nb_classes)


dataX,dataY = create_dataset()

print dataX.shape
print dataY.shape

class EarlyStopping(Callback):
    def __init__(self, monitor='accuracy', value=1.0, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print ("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


model = Sequential()

model.add(Dense(64, input_shape=(10,)))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(4))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


callbacks = [EarlyStopping(monitor='loss',value=0.01,verbose=1)]

model.fit(dataX,dataY,nb_epoch=10000,batch_size=batch_size,callbacks=callbacks)

for i in range(1,101):
    x = bin_encode(i)
    y = model.predict(numpy.array(x).reshape(-1,10))
    print fizz_buzz(i,y)

