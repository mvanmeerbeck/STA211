from keras.datasets import mnist

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)

# import matplotlib as mpl
# mpl.use('TKAgg')
# import matplotlib.pyplot as plt
# plt.figure(figsize=(7.195, 3.841), dpi=100)
# for i in range(200):
#     plt.subplot(10,20,i+1)
#     plt.imshow(X_train[i,:].reshape([28,28]), cmap='gray')
#     plt.axis('off')
# plt.show()

from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

model.add(Conv2D(32, kernel_size=(5, 5), activation='sigmoid', input_shape=(28, 28, 1), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5, 5), activation='sigmoid', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('sigmoid'))
model.add(Dense(10))
model.add(Activation('softmax'))

from keras.optimizers import SGD

sgd = SGD(lr=0.01, momentum=0, decay=0, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

from keras.utils import np_utils

batch_size = 300
nb_epoch = 10
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)
model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1)

scores = model.evaluate(X_test, Y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[0], scores[0] * 100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

model.save('MLP.h5')

yaml_string = model.to_yaml()
print(yaml_string)