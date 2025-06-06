import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# x是输入 rate是比例 pt是高频或者低频
def fft(x, rate, prompt_type):
    mask = torch.zeros(x.shape)
    w, h = x.shape[-2:]
    line = int((w * h * rate) ** .5 // 2)  # 中轴线
    mask[:, w // 2 - line:w // 2 + line, h // 2 - line:h // 2 + line] = 1  # 生成mask

    # 傅里叶变换
    fft0 = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))

    if prompt_type == 'highpass':
        fft = fft0 * (1 - mask)
    elif prompt_type == 'lowpass':
        fft = fft0 * mask

    fr = fft.real  # 实部 表示振幅
    fi = fft.imag  # 虚部 表示相位

    visualize_spectrum(fft)
    # 把振幅和相位融合
    fft_hires = torch.fft.ifftshift(torch.complex(fr, fi))
    # 傅里叶变换的逆
    inv = torch.fft.ifft2(fft_hires, norm="forward").real
    inv = torch.abs(inv)
    return fft0, inv


def visualize_spectrum(fft):
    magnitude = torch.abs(fft)
    # magnitude = torch.log1p(magnitude)  # 取对数以更好地可视化

    # magnitude = torch.clamp(magnitude, max=0.001)
    # 将张量转换为NumPy数组
    magnitude_np = magnitude.cpu().numpy()
    # 如果输入是批次数据，则只取第一个样本进行可视化
    if len(magnitude_np.shape) == 3:
        magnitude_np = magnitude_np[0, :, :]+magnitude_np[1, :, :]+magnitude_np[2, :, :]

    plt.figure(figsize=(10, 10))
    plt.imshow(magnitude_np, cmap='gray')
    plt.title('Frequency Spectrum')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    # 读取彩色图像
    image = Image.open(r"./img_2.png")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # 调整图像尺寸为256x256
        transforms.ToTensor()  # 将图像转换为张量
    ])

    # 对图像应用转换操作
    tensor_image = transform(image)
    fft_result, HF_image = fft(tensor_image, 0.02, "highpass")

    # 可视化FFT结果
    visualize_spectrum(fft_result)

    # 可视化高频图像
    HF_image = np.transpose(HF_image.numpy(), (1, 2, 0))
    plt.imshow(HF_image)
    plt.axis('off')  # 关闭坐标轴
    plt.show()
    print("end")
