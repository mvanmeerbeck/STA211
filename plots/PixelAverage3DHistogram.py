from matplotlib import rcParams
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

rcParams['font.family'] = 'EB Garamond'
rcParams['font.size'] = '13'

mnist = mnist.load_data()

# setup the figure and axes
fig = plt.figure()
ax = Axes3D(fig)

_x = np.arange(28)
_y = np.arange(28)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = 255
bottom = np.zeros_like(top)
width = depth = 1

ax.set_xlim(0,28)
ax.set_ylim(28,0)

top = mnist[0][0].mean(axis=1, keepdims=False).reshape(1, 784)

ax.bar3d(x, y, bottom, width, depth, top[0], shade=True)
ax.set_title('Shaded')

plt.show()