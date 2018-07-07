import matplotlib.pyplot as plt

from matplotlib import rcParams
from sklearn import datasets
from sklearn.decomposition import PCA

#rcParams['font.family'] = 'EB Garamond'
rcParams['font.size'] = '13'

digits = datasets.load_digits()

X = digits.data
y = digits.target
target_names = digits.target_names

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

plt.figure()
colors = ['navy', 'turquoise', 'darkorange', 'blue', 'red', 'green', 'pink', 'black', 'yellow', 'grey']
lw = 2

for color, i, target_name in zip(colors, target_names, target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('ACP : Les chiffres sur le plan factoriel')

plt.show()