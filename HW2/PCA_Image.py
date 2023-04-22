import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from sklearn.decomposition import PCA

A = io.imread('test.jpg')
A = A / 255  # RGB的三个值[0,255]，将它们范围设置为[0,1]

R, G, B = A[:, :, 0], A[:, :, 1], A[:, :, 2]


def pca(X, K):
    pca = PCA(n_components=K).fit(X)
    X_new = pca.transform(X)
    X_new = pca.inverse_transform(X_new)
    return X_new


ratio = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]
index = []
for i in ratio:
    index.append(int(555 * i))

for k in index:
    R_new, G_new, B_new = pca(R, k), pca(G, k), pca(B, k)

    A_new = np.zeros(A.shape)
    A_new[:, :, 0] = R_new
    A_new[:, :, 1] = G_new
    A_new[:, :, 2] = B_new

    fig, ax_array = plt.subplots(nrows=1, ncols=1, figsize=(64, 64))
    cmap_list = ['Reds', 'Greens', 'Blues']
    ax_array.imshow(A_new[:, :, :])
    ax_array.set_xticks([])
    ax_array.set_yticks([])
    ax_array.set_title("Image", size=30, color='g')
    plt.show()
