import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def creating_new_s(s, dim1, dim2, cutoff):
    clean_s = np.zeros((dim1, dim2))
    for k in range(cutoff):
        clean_s[k, k] = s[k]
    return clean_s


def get_clean_matrix(mat, dim1, dim2, cutoff):
    u, s, v = np.linalg.svd(mat)
    sigma = creating_new_s(s, dim1, dim2, cutoff)
    return u @ sigma @ v


# Load the image
image = plt.imread('noisy.jpg')
d1 = image.shape[0]
d2 = image.shape[1]
img = image / 255


# Getting the three matrices
red = img[:, :, 0]
green = img[:, :, 1]
blue = img[:, :, 2]

# I plot the s and I saw after the almost 20th singular value, it gets pretty small. so, I set the cutoff to 20
red_new = get_clean_matrix(red, d1, d2, cutoff=20)
green_new = get_clean_matrix(green, d1, d2, cutoff=20)
blue_new = get_clean_matrix(blue, d1, d2, cutoff=20)

# combining the three rgb matrices to get the final matrix
cleaned_matrix = np.zeros((d1, d2, 3))
cleaned_matrix[:, :, 0] = red_new
cleaned_matrix[:, :, 1] = green_new
cleaned_matrix[:, :, 2] = blue_new

print(type(cleaned_matrix))
print(cleaned_matrix.shape)
plt.imshow(cleaned_matrix)
plt.show()
plt.imsave("denoisy.jpeg", (cleaned_matrix * 255).astype(np.uint8))