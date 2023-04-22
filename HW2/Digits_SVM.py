from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm
import matplotlib.pyplot as plt
digits = load_digits()
# 原矩阵为8*8
# PCA压缩前分类
Xtrain, Xtest, Ytrain, Ytest = train_test_split(digits.data, digits.target, test_size=0.25, random_state=114514)
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(Xtrain, Ytrain)
score1 = clf.score(Xtest,Ytest)

fig, axes = plt.subplots(4, 4, figsize=(8,8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

Ypred = clf.predict(Xtest)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.05, 0.05, str(Ypred[i]), fontsize=32, transform=ax.transAxes, color='green' if Ypred[i] == Ytest[i] else 'red')
    ax.text(0.8, 0.05, str(Ytest[i]), fontsize=32, transform=ax.transAxes, color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()
print("PCA压缩前测试集分类正确率:",score1)

# PCA压缩后分类

pca = PCA(n_components=32)
X_trans_train = pca.fit_transform(Xtrain)
X_trans_test = pca.fit_transform(Xtest)
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_trans_train, Ytrain)
score2 = clf.score(X_trans_test,Ytest)

fig, axes = plt.subplots(4, 4, figsize=(8,8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)

Ypred = clf.predict(X_trans_test)

for i, ax in enumerate(axes.flat):
    ax.imshow(Xtest[i].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(0.05, 0.05, str(Ypred[i]), fontsize=32, transform=ax.transAxes, color='green' if Ypred[i] == Ytest[i] else 'red')
    ax.text(0.8, 0.05, str(Ytest[i]), fontsize=32, transform=ax.transAxes, color='black')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

print("PCA压缩后测试集分类正确率:",score2)

