import numpy as np
import matplotlib.pyplot as plt

def generate_mask(W, H, sigma):
    i_coords = np.arange(W).reshape(-1, 1)  # W维列向量
    j_coords = np.arange(H).reshape(1, -1)  # H维行向量

    # 中心坐标
    center_i = W / 2
    center_j = H / 2

    # 高斯反函数
    Mh =np.exp(-((i_coords - center_i) ** 2 + (j_coords - center_j) ** 2) / (2 * sigma ** 2))
    return Mh

# 参数设置
W, H = 240, 240
sigma = 10

# 生成 mask
mask = generate_mask(W, H, sigma)

# plt.imsave('mask.png', mask)
# 可视化
plt.figure(figsize=(6, 5))
plt.imshow(mask.T, cmap='binary', origin='lower')  # 注意转置一下让方向对齐
plt.colorbar(label='Mask Value')
plt.title('Visualization of $M_h^{i,j}$')
plt.xlabel('i (Width)')
plt.ylabel('j (Height)')
plt.tight_layout()
plt.show()
