from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

mnist_images = mnist.train.images
print(mnist_images)
print(mnist)


mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data, mnist.target
X = X.astype('float32')
X /= 255

clf_rf = KNeighborsClassifier()

cv = ShuffleSplit(n_splits=1, test_size=1/7, random_state=0)
scores = cross_val_score(clf_rf, X, y, cv=cv)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))