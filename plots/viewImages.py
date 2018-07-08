import matplotlib as mpl
mpl.use('TKAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from matplotlib import rcParams

rcParams['font.family'] = 'EB Garamond'
rcParams['font.size'] = '13'

mnist = load_digits()

X, y = mnist.data, mnist.target

plt.figure(figsize=(7.195, 3.841), dpi=100)
for i in range(100):
    plt.subplot(10,10,i+1)
    plt.imshow(X[i,:].reshape([8,8]), cmap='gray')
    plt.axis('off')
plt.show()