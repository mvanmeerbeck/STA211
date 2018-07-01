from keras.utils import np_utils
from sklearn.model_selection import ShuffleSplit, LeaveOneOut

print(__doc__)

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data, mnist.target
X = X.astype('float32')
X /= 255

X = X.reshape(X.shape[0], 28, 28, 1)

# X_train, X_test = X[:60000], X[60000:]
# y_train, y_test = y[:60000], y[60000:]

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
# mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=10, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1,
#                     learning_rate_init=.1)
#
# mlp.fit(X_train, y_train)
# print("Training set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))

from sklearn import tree

clf = tree.DecisionTreeClassifier()

# clf.fit(X_train, y_train)
# print("Training set score: %f" % clf.score(X_train, y_train))
# print("Test set score: %f" % clf.score(X_test, y_test))

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD


def create_model():
    model = Sequential()

    # model.add(Conv2D(32, kernel_size=(5, 5), activation='sigmoid', input_shape=(28, 28, 1), padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(64, kernel_size=(5, 5), activation='sigmoid', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(100, activation='sigmoid'))
    # model.add(Dense(10, activation='softmax'))

    model.add(Conv2D(20, 5, padding="same",
                     input_shape=(28, 28, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(50, 5, padding="same"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation('relu'))
    # define the second FC layer
    model.add(Dense(10))

    # lastly, define the soft-max classifier
    model.add(Activation('softmax'))


    sgd = SGD(lr=0.01, momentum=0, decay=0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model


clf = KerasClassifier(build_fn=create_model,
                      epochs=10,
                      batch_size=300,
                      verbose=1)

y = np_utils.to_categorical(y, 10)

from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=1, test_size=1/7, random_state=0)
scores = cross_val_score(clf, X, y, cv=cv)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# import graphviz
# dot_data = tree.export_graphviz(clf, out_file=None,
#                          filled=True, rounded=True,
#                          special_characters=True)
#
# graph = graphviz.Source(dot_data)
# graph.format = 'svg'
# graph.render('tree')
