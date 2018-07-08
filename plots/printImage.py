import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from matplotlib import rcParams

rcParams['font.family'] = 'EB Garamond'
rcParams['font.size'] = '13'



mnist = load_digits()

X, y = mnist.data, mnist.target
X = X.astype('int')

print(X[0])

X = X.reshape(X.shape[0], 8, 8, 1)

print(X[0])

