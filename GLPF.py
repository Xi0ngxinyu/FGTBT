import torch
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np


class FFTFilter:
    def fft(self, x, sigma, prompt_type):
        device = x.device
        b, c, h, w = x.shape

        # 生成坐标网格，默认采用 ij 索引方式
        y = torch.arange(h, device=device) - (h // 2)
        x_coords = torch.arange(w, device=device) - (w // 2)
        y_grid, x_grid = torch.meshgrid(y, x_coords)
        D_squared = x_grid**2 + y_grid**2

        # 生成高斯低通 mask 及其对应的高通 mask
        lowpass_mask = torch.exp(-D_squared / (2 * (sigma ** 2)))
        highpass_mask = 1 - lowpass_mask

        if prompt_type == 'highpass':
            mask = highpass_mask
        elif prompt_type == 'lowpass':
            mask = lowpass_mask
        else:
            raise ValueError("prompt_type 必须为 'highpass' 或 'lowpass'")

        # 扩展 mask 维度以匹配 x 的 shape
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
        # 傅里叶变换（前向，归一化，并中心化）
        fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"), dim=(-2, -1))
        # 应用频域滤波器
        fft_filtered = fft * mask
        # 逆中心化并计算逆傅里叶变换
        fft_filtered = torch.fft.ifftshift(fft_filtered, dim=(-2, -1))
        inv = torch.fft.ifft2(fft_filtered, norm="forward").real
        inv = torch.abs(inv)
        return inv


def generate_mask(W, H, sigma):
    i_coords = np.arange(W).reshape(-1, 1)  # W维列向量
    j_coords = np.arange(H).reshape(1, -1)  # H维行向量

    # 中心坐标
    center_i = W / 2
    center_j = H / 2

    # 高斯反函数
    Mh = 1 - np.exp(-((i_coords - center_i) ** 2 + (j_coords - center_j) ** 2) / (2 * sigma ** 2))
    return torch.tensor(Mh)


def visualize_spectrum():

    image = Image.open(r"./1.jpg")
    transform = transforms.Compose([
        transforms.Resize((240, 240)),  # 调整图像尺寸为256x256
        transforms.ToTensor()  # 将图像转换为张量
    ])

    # 对图像应用转换操作
    tensor_image = transform(image)

    s = torch.fft.fftshift(torch.fft.fft2(tensor_image, norm="forward"), dim=(-2, -1))
    magnitude = torch.abs(s)
    magnitude = torch.clamp(magnitude, max=0.005)
    mask = generate_mask(240, 240, 10)
    magnitude_mask = mask*magnitude
    # magnitude = torch.log1p(magnitude)  # 取对数以更好地可视化

    # 将张量转换为NumPy数组
    magnitude_np = magnitude_mask.cpu().numpy()
    # 如果输入是批次数据，则只取第一个样本进行可视化
    if len(magnitude_np.shape) == 3:
        magnitude_np = magnitude_np[0, :, :]+magnitude_np[1, :, :]+magnitude_np[2, :, :]
    # plt.imsave('spectrum.png', magnitude_np)
    plt.figure(figsize=(10, 10))
    plt.imshow(magnitude_np, cmap='jet')
    plt.title('Frequency Spectrum')
    plt.colorbar()
    plt.show()


# 用于增强高频图像可视化效果
def enhance_image_for_display(img_tensor):
    # img_tensor: shape [B, C, H, W]
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # 归一化到0~255范围
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0
    return img_np.astype('uint8')


# 示例调用：
filter_instance = FFTFilter()

# 读取单张图片（BGR 格式），并转换为 torch.Tensor
img_bgr = cv2.imread(r"1.jpg")
if img_bgr is None:
    raise ValueError("图像加载失败，请检查路径！")

# 将 BGR 转换为 RGB 便于显示
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

visualize_spectrum()

# 将图像数据转换为 float32，并转换为 [C, H, W] 格式，再扩展 batch 维度
img_tensor = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

sigma = 20  # 控制滤波器宽度
output = filter_instance.fft(img_tensor, sigma, 'highpass')
output_np = enhance_image_for_display(output)
# 将结果转换为 numpy 格式以便显示
# output_np = output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype('uint8')

img_np = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype('uint8')

plt.imsave('filtered_image.png', output_np)
# 可视化原始图像和滤波后图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title('原始图像')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(output_np)
plt.title('高斯高通滤波后')
plt.axis('off')
plt.show()









