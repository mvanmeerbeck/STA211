from sklearn.datasets import fetch_mldata
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit

mnist = fetch_mldata("MNIST original")
X, y = mnist.data, mnist.target
X = X.astype('float32')
X /= 255

clf_rf = SGDClassifier()

cv = ShuffleSplit(n_splits=5, test_size=1/7, random_state=0)
scores = cross_val_score(clf_rf, X, y, cv=cv)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))