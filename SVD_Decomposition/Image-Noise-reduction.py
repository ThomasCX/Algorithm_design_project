from numpy import *
from numpy import linalg as la
from PIL import Image
import matplotlib.pyplot as plt

path = 'C:/Users/10741/Desktop/svd算法作业/雀斑girl.dib'
image = Image.open(path).convert('L')
image = array(image)
print(shape(image))
plt.figure(1)
plt.imshow(image, cmap='gray')


# 降噪方法
def svd_denoise(image, k):
    u, sigma, vt = la.svd(image)
    h, w = image.shape[:2]
    # 取前10%的奇异值重构图像
    h1 = int(h * k)
    # 用奇异值生成对角矩阵
    sigma1 = diag(sigma[:h1], 0)
    u1 = zeros((h, h1), float)
    u1[:, :] = u[:, :h1]
    vt1 = zeros((h1, w), float)
    vt1[:, :] = vt[:h1, :]
    a = u1.dot(sigma1).dot(vt1)
    plt.imshow(a, cmap='gray')
    plt.title('k=%s' % k)


plt.figure(2)
svd_denoise(image, 0.01)

plt.figure(3)
svd_denoise(image, 0.05)

plt.figure(4)
svd_denoise(image, 0.1)

plt.show()
