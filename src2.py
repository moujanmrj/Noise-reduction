from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import math


def get_main_function():
    x = np.linspace(0, math.pi * 3 / 2, 30)
    y = np.linspace(0, math.pi * 3 / 2, 30)
    X, Y = np.meshgrid(x, y)
    return np.sin(X * Y)


def make_function_noisy(z):
    max_noise = 0.1
    t = 2 * np.random.rand(30 * 30) * max_noise - max_noise
    t = t.reshape(30, 30)
    return np.add(z, t)


def get_function():
    return make_function_noisy(get_main_function())


def show_my_matrix(Z):
    x = np.linspace(0, math.pi * 3 / 2, 30)
    y = np.linspace(0, math.pi * 3 / 2, 30)
    X, Y = np.meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='viridis', edgecolor='black')
    ax.set_title('surface')
    ax.view_init(60, 35)
    plt.show()


def show_main():
    Z = get_main_function()
    show_my_matrix(Z)


def show_noisy():
    Z = get_function()
    show_my_matrix(Z)

# I found the k = 9, suitable for the cutoff index for singular values.


if __name__ == '__main__':
    # show_main()
    # show_noisy()
    z = get_function()
    U, S, V = np.linalg.svd(z)
    print(S.shape)
    print(S)
    print(V.shape)
    print(U.shape)
    Sigma = np.zeros((30, 30))
    for k in range(9):
        Sigma[k, k] = S[k]
    # plt.plot(S)
    # plt.show()
    cleaned_matrix = U @ Sigma @ V
    show_my_matrix(cleaned_matrix)
