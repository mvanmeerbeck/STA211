import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import datasets
from sklearn.manifold import TSNE

#rcParams['font.family'] = 'EB Garamond'
rcParams['font.size'] = '13'

digits = datasets.load_digits()

X = digits.data
y = digits.target
target_names = digits.target_names

tsne = TSNE(init='pca')
X_r = tsne.fit_transform(X)

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'blue', 'red', 'green', 'pink', 'black', 'yellow', 'grey']
lw = 2

for color, i, target_name in zip(colors, target_names, target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('2D t-SNE')

plt.show()