import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False
n_list = [0, 1, 2, 3, 8, 15, 50, 500]
x_list = [0, 1, 2, 2, 4, 6, 24, 263]


def draw(ax, a, b, n, x):
    post_a = a + x
    post_b = b + n - x
    x_line = np.linspace(beta.ppf(0.001, post_a, post_b), beta.ppf(0.999, post_a, post_b), 1000)
    ax.plot(x_line, beta.pdf(x_line, post_a, post_b), lw=2,
            label="toss a coin {n} times with {x} heads".format(n=n, x=x))
    ax.legend(loc=(0.6,0.6))
    plt.xlim(0, 1)


plt.figure(figsize=(15, 10))
ax = plt.subplot(1, 2, 1)
ax.set_title("Beta(1,1)")
for n, x in zip(n_list, x_list):
    draw(ax, 1, 1, n, x)
ax = plt.subplot(1, 2, 2)
ax.set_title("Beta(10,5)")
for n, x in zip(n_list, x_list):
    draw(ax, 10, 5, n, x)
plt.show()
