from sklearn.datasets import fetch_mldata
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_mldata("MNIST original")
# rescale the data, use the traditional train/test split
X, y = mnist.data, mnist.target
X = X.astype('float32')
X /= 255

clf_rf = KernelRidge()

cv = ShuffleSplit(n_splits=1, test_size=1/7, random_state=0)
scores = cross_val_score(clf_rf, X, y, cv=cv)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))