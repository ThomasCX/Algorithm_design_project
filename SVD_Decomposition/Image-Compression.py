import numpy as np
from numpy.linalg import svd
from PIL import Image
import matplotlib.pyplot as plt

path = 'C:/Users/10741/Desktop/svd算法作业/tf0.jpg'
image = Image.open(path).convert('L')
image = np.array(image)
print(np.shape(image))
origin_size = image.shape[0] * image.shape[1]
print('原图像大小为:%d' % origin_size)
plt.figure(1)
plt.imshow(image, cmap='gray')
plt.savefig('C:/Users/10741/Desktop/svd算法作业/tf1.jpg')


# 下面分别对RGB图像做SVD压缩
def svd(matrix, k):
    # 奇异值分解
    u, z, vt = np.linalg.svd(matrix)
    # 取前k个奇异值
    u = u[:, 0:k]
    z = np.diag(z[0:k])
    vt = vt[0:k, :]
    # 重新组合图片
    a = u.dot(z).dot(vt)
    # 把不正常的颜色变为正常
    a[a < 0] = 0
    a[a > 255] = 255
    matrix[:, :] = a
    size = u.shape[0] * k + z.shape[0] * z.shape[1] + k * vt.shape[1]
    return size, matrix


def compression(image, k):
    compress_size, image = svd(image, k)
    plt.title('k=%s' % k)
    plt.imshow(image, cmap='gray')
    print('选择%d个奇异值后的压缩尺寸：%d' % (k, compress_size))


plt.figure(figsize=(20, 10))
k = 128
for i in range(8):
    k = int(k/2)
    pos = 241+i
    plt.subplot(pos)
    compression(image, k)
plt.show()
