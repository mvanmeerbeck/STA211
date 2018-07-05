import matplotlib.pyplot as plt
from matplotlib import rcParams
from keras.datasets import mnist

rcParams['font.family'] = 'EB Garamond'
rcParams['font.size'] = '13'

mnist = mnist.load_data()

plt.plot()
plt.hist(mnist[0][0][0], 10, histtype='bar', stacked=True, facecolor='#c1002a')

plt.axes().set_ylabel('Nombre de pixels')
plt.axes().set_xlabel('Valeur du pixel')
plt.title('Distribution des valeurs des pixels')

plt.show()