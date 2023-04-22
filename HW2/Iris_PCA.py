from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

iris = load_iris()
# 确定函数和函数方法
pca = PCA(n_components=3)
data = pca.fit_transform(iris.data)

# 取出各类样本对应样本
setosa = np.where(iris.target == 0)
versicolor = np.where(iris.target == 1)
virginica = np.where(iris.target == 2)
labels = ["setosa", "versicolor", "virginica"]


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(data[setosa][:, 0], data[setosa][:, 1], data[setosa][:, 2],c = 'r')
ax.scatter(data[versicolor][:, 0], data[versicolor][:, 1], data[versicolor][:, 2],c = 'g')
ax.scatter(data[virginica][:, 0], data[virginica][:, 1], data[virginica][:, 2],c = 'b')
ax.set_xlabel('Component1')
ax.set_ylabel('Component2')
ax.set_zlabel('Component3')
plt.show()
