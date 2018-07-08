import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.datasets import load_digits

rcParams['font.family'] = 'EB Garamond'
rcParams['font.size'] = '13'

digits = load_digits()

X, y = digits.data, digits.target

plt.plot()
plt.hist(X[1], 10, histtype='bar', stacked=True, facecolor='#c1002a')

plt.axes().set_ylabel('Nombre de pixels')
plt.axes().set_xlabel('Valeur du pixel')
plt.title('Distribution des valeurs des pixels')

plt.show()