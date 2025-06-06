from datasets.transforms import get_transformer_coords
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np


def generate_mask(W, H, sigma):
    i_coords = np.arange(W).reshape(-1, 1)  # W维列向量
    j_coords = np.arange(H).reshape(1, -1)  # H维行向量

    # 中心坐标
    center_i = W / 2
    center_j = H / 2

    # 高斯反函数
    Mh = 1 - np.exp(-((i_coords - center_i) ** 2 + (j_coords - center_j) ** 2) / (2 * sigma ** 2))
    return torch.tensor(Mh)


def visualize_spectrum(image, save_dir, base_name):
    # image = Image.open(r"./img_2.png")
    # 第一步：BGR -> RGB
    image = image[:, :, ::-1]
    # 第二步：从 NumPy 数组创建 PIL 图像
    image = Image.fromarray(image)
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

    ###############################################################################
    # 保存“乘 mask 之前”的频谱图像
    magnitude_np_before = magnitude.cpu().numpy()
    # 如果是多通道，则按通道求和
    if magnitude_np_before.ndim == 3:
        # C 通道相加得到二维图
        magnitude_before = magnitude_np_before[0] + magnitude_np_before[1] + magnitude_np_before[2]
    else:
        magnitude_before = magnitude_np_before
    # 归一化到 [0, 255]
    mag_before_norm = (magnitude_before - magnitude_before.min()) / (magnitude_before.max() - magnitude_before.min())
    mag_before_img = (mag_before_norm * 255).astype(np.uint8)
    # 保存为 PNG
    plt.imsave(os.path.join(save_dir,  f'{base_name}_spactrum_before.png'),
               mag_before_img,
               cmap='jet')
    ############################################################################

    magnitude_mask = mask*magnitude
    # magnitude = torch.log1p(magnitude)  # 取对数以更好地可视化

    # 将张量转换为NumPy数组
    magnitude_np = magnitude_mask.cpu().numpy()

    if magnitude_np.ndim == 3:
        magnitude_after = magnitude_np[0] + magnitude_np[1] + magnitude_np[2]
    else:
        magnitude_after = magnitude_np
    mag_after_norm = (magnitude_after - magnitude_after.min()) / (magnitude_after.max() - magnitude_after.min())
    mag_after_img = (mag_after_norm * 255).astype(np.uint8)
    plt.imsave(os.path.join(save_dir, f'{base_name}_spactrum_after.png'),
               mag_after_img,
               cmap='jet')

    # 如果输入是批次数据，则只取第一个样本进行可视化
    # if len(magnitude_np.shape) == 3:
    #     magnitude_np = magnitude_np[0, :, :]+magnitude_np[1, :, :]+magnitude_np[2, :, :]
    # # plt.imsave('spectrum.png', magnitude_np)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(magnitude_np, cmap='jet')
    # plt.title('Frequency Spectrum')
    # plt.colorbar()
    # plt.show()


def insert_lists_to_excel(file_path, list1, list2, start_col=0, start_row=0):
    try:
        df = pd.read_excel(file_path, header=None)
    except FileNotFoundError:
        df = pd.DataFrame()

    max_rows = start_row + max(len(list1), len(list2))
    max_cols = start_col + 2  # 需要两列

    # 扩展行
    if df.shape[0] < max_rows:
        df = df.reindex(range(max_rows))

    # 扩展列
    if df.shape[1] < max_cols:
        for _ in range(max_cols - df.shape[1]):
            df[df.shape[1]] = np.nan

    df.iloc[start_row:start_row + len(list1), start_col] = list1
    df.iloc[start_row:start_row + len(list2), start_col + 1] = list2

    df.to_excel(file_path, index=False, header=False)


def visualize_with_landmarks(images, predict, targets):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch_size = images.shape[0]

    for i in range(batch_size):
        image = images[i]
        gt_landmarks = targets[i]  # 真实值
        pred_landmarks = predict[i]  # 预测值

        # 将 PyTorch 张量转换为 NumPy 数组
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))  # 从 [C, H, W] 转换为 [H, W, C]

        # 反标准化图像
        image_np = (image_np * std + mean) * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)

        # 转换为 BGR（OpenCV 使用 BGR 顺序）
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_with_landmarks = np.ascontiguousarray(image_bgr)

        # 画真实值
        for landmark in gt_landmarks:
            x, y = landmark[0] * 2, landmark[1] * 2
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)
        # 画预测值
        for landmark in pred_landmarks:
            x, y = landmark[0] * 2, landmark[1] * 2
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)

        # 显示图像
        plt.imshow(cv2.cvtColor(image_with_landmarks, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

class FFTFilter:
    def fft(self, x: torch.Tensor, sigma: float, prompt_type: str, save_dir) -> torch.Tensor:
        """
        对单张或批量图像做 FFT 域高/低通滤波
        Args:
            x: [B, C, H, W] 输入张量
            sigma: 高斯滤波标准差
            prompt_type: 'highpass' 或 'lowpass'
        Returns:
            [B, C, H, W] 滤波后张量（实部取 abs）
        """
        b, c, h, w = x.shape
        device = x.device

        # 构造频域坐标
        y = torch.arange(h, device=device) - (h // 2)
        x_ = torch.arange(w, device=device) - (w // 2)
        yv, xv = torch.meshgrid(y, x_,)
        D2 = xv**2 + yv**2

        lowpass = torch.exp(-D2 / (2 * sigma**2))
        highpass = 1 - lowpass
        if prompt_type == 'highpass':
            mask = highpass
        elif prompt_type == 'lowpass':
            mask = lowpass
        else:
            raise ValueError("prompt_type 必须为 'highpass' 或 'lowpass'")
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        # FFT -> shift -> 乘 mask -> ifftshift -> iFFT
        Xf = torch.fft.fftshift(torch.fft.fft2(x, norm='forward'), dim=(-2, -1))
        # 应用 mask
        Xf = Xf * mask
        Xf = torch.fft.ifftshift(Xf, dim=(-2, -1))
        x_rec = torch.fft.ifft2(Xf, norm='forward').real
        return torch.abs(x_rec)


def enhance_image_for_display(img_tensor):
    # img_tensor: shape [B, C, H, W]
    img_np = img_tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    # 归一化到0~255范围
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min()) * 255.0
    return img_np.astype('uint8')


# 提取出的边界
# 保存预测值点打在图像上 真实值 预测的损失NME写在文件名中
def save_landmarks_with_nme(images, predicts, targets, filenames, save_dir='output_images', nmes=None):
    """
    保存两张图像：
      1. 真实值的标记图（绿色点）
      2. 预测值的标记图（红色点），文件名包含 NME

    参数:
        images: torch.Tensor, [B, C, H, W]
        predicts: array-like, [B, num_landmarks, 2]
        targets: array-like, [B, num_landmarks, 2]
        filenames: list of str, 每张图像的原始文件名（含扩展名）
        save_dir: str, 输出目录
        nmes: list of float, 长度 B，对应每张图的 NME 值
    """
    # 还原图像的均值和方差
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    os.makedirs(save_dir, exist_ok=True)
    batch_size = images.shape[0]
    nmes = nmes if nmes is not None else [''] * batch_size

    for i in range(batch_size):
        # 图像反归一化到像素空间
        img = images[i].cpu().float().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img * std + mean) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if isinstance(filenames, torch.Tensor):
            base = filenames[i]
        else:
            base = os.path.splitext(os.path.basename(filenames[i]))[0][-20:]
        visualize_spectrum(img_bgr, save_dir, base)

        # 文件名基准
        nme_val = nmes[i]
        nme_str = f"_NME_{nme_val:.4f}"

        # 1. 保存真实值图像
        gt_img = img_bgr.copy()
        # gt0_path = os.path.join(save_dir, f"{base}_gt0.jpg")
        # cv2.imwrite(gt0_path, gt_img)
        # # 保存打了点的
        for (x, y) in predicts[i]:
            px, py = int(x * 2), int(y * 2)
            cv2.circle(gt_img, (px, py), radius=5, color=(0, 255, 0), thickness=-1)
        gt_path = os.path.join(save_dir, f"{base}_predict.jpg")
        cv2.imwrite(gt_path, gt_img)
        #
        # # 2. 保存预测值图像
        pred_img = img_bgr.copy()
        for (x, y) in targets[i]:
            px, py = int(x * 2), int(y * 2)
            cv2.circle(pred_img, (px, py), radius=3, color=(0, 255, 0), thickness=-1)
        for (x, y) in predicts[i]:
            px, py = int(x * 2), int(y * 2)
            cv2.circle(pred_img, (px, py), radius=4, color=(0, 0, 255), thickness=-1)
        pred_path = os.path.join(save_dir, f"{base}{nme_str}_pred_bam.jpg")
        cv2.imwrite(pred_path, pred_img)
        # # 3. FFT 滤波并保存
        # 将反归一化后的 img 转回 Tensor [1,C,H,W]
        img_tensor = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).float()
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
        filterer = FFTFilter()
        filtered = filterer.fft(img_tensor, 20, "highpass", save_dir)
        arr = enhance_image_for_display(filtered)
        # 归一化并转 uint8
        # arr = filtered.squeeze(0).permute(1, 2, 0).numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255.0
        filt_img = arr.astype(np.uint8)
        filt_bgr = cv2.cvtColor(filt_img, cv2.COLOR_RGB2BGR)
        filt_path = os.path.join(save_dir, f"{base}_prompt.jpg")
        cv2.imwrite(filt_path, filt_bgr)
    # print(f"Saved {batch_size * 2} images (GT and pred) to '{save_dir}'.")


def save_images_with_landmarks(images, predict, targets, filenames, save_dir='output_images'):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    batch_size = images.shape[0]
    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)
    for i in range(batch_size):
        image = images[i]
        gt_landmarks = targets[i]
        pred_landmarks = predict[i]
        image_np = image.cpu().numpy()
        image_np = np.transpose(image_np, (1, 2, 0))
        image_np = (image_np * std + mean) * 255.0
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        image_with_landmarks = np.ascontiguousarray(image_bgr)
        # 创建一个白色背景图像（尺寸放大2倍，因为关键点乘了2）
        # h, w = 240, 240
        # image_with_landmarks = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255  # 白底

        # 画真实值（绿色）
        # for landmark in gt_landmarks:
        #     x, y = landmark[0] * 2, landmark[1] * 2
        #     cv2.circle(image_with_landmarks, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)

        # filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        filename = filenames[i]
        save_path = os.path.join(save_dir, f'test_{filename}_0.jpg')
        cv2.imwrite(save_path, image_with_landmarks)

        # 画预测值（红色）
        for landmark in pred_landmarks:
            x, y = landmark[0] * 2, landmark[1] * 2
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=4, color=(0, 255, 0), thickness=-1)
        # 可选：在图像左上角添加说明文字
        # cv2.putText(image_with_landmarks, 'Green: GT, Red: Pred', (10, 25),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # 保存图像
        save_path = os.path.join(save_dir, f'test_{filename}.jpg')
        cv2.imwrite(save_path, image_with_landmarks)


def get_points(cfg, images, predict, filenames, save_dir='output_images'):
    indexes = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 0~9
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10~19
        1, 1, 1, 1, 4, 1, 1, 1, 1, 1,  # 20~29
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 30~39
        1, 1, 1, 1, 1, 1, 1, 1, 1, 4,  # 40~49
        2, 4, 2, 4, 1, 1, 2, 1, 4, 2,  # 50~59
        4, 2, 4, 1, 2, 1, 1, 2, 2, 2,  # 60~69
        4, 2, 2, 3, 2, 2, 2, 2, 4, 1,  # 70~79
        1, 2, 1, 1, 4, 1, 1, 2, 1, 1,  # 80~89
        4, 1, 1, 2, 1, 1, 4, 1, 1, 2,  # 90~99
        1, 1, 4, 2, 2, 3, 2, 2, 4, 2,  # 100~109
        2, 3, 2, 2, 2, 2, 3, 2, 2, 2,  # 110~119
        4, 2, 3, 3]
    colors = [(0, 255, 0), (255, 0, 0), (0, 200, 255), (0, 0, 255)]
    predict = get_preds(predict)
    batch_size = images.shape[0]
    # 创建保存文件夹
    os.makedirs(save_dir, exist_ok=True)
    for i in range(batch_size):
        pred_landmarks = predict[i]
        # 创建一个白色背景图像
        h, w = 240, 240
        image_with_landmarks = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255  # 白底
        # 画预测值（红色）
        for j in range(124):
            landmark = pred_landmarks[j]
            x, y = landmark[0] * 2, landmark[1] * 2
            cv2.circle(image_with_landmarks, (int(x), int(y)), radius=4, color=colors[indexes[j]-1], thickness=-1)
        # 保存图像
        filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        save_path = os.path.join(save_dir, f'test_{filename}_0.jpg')
        cv2.imwrite(save_path, image_with_landmarks)
        #
        image_with_landmarks1 = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255  # 白底
        # 画预测值
        for j in range(124):
            if j in cfg.DATASET_300W.LANDMARK_INDEX:
                landmark = pred_landmarks[j]
                x, y = landmark[0] * 2, landmark[1] * 2
                cv2.circle(image_with_landmarks1, (int(x), int(y)), radius=4, color=colors[indexes[j]-1], thickness=-1)
        # 保存图像
        filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        save_path = os.path.join(save_dir, f'test_{filename}_1.jpg')
        cv2.imwrite(save_path, image_with_landmarks1)

        image_with_landmarks2 = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255  # 白底
        # 画预测值（红色）
        for j in range(124):
            if j in cfg.DATASET_WFLW.LANDMARK_INDEX:
                landmark = pred_landmarks[j]
                x, y = landmark[0] * 2, landmark[1] * 2
                cv2.circle(image_with_landmarks2, (int(x), int(y)), radius=4, color=colors[indexes[j]-1], thickness=-1)
        # 保存图像
        filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        save_path = os.path.join(save_dir, f'test_{filename}_2.jpg')
        cv2.imwrite(save_path, image_with_landmarks2)

        image_with_landmarks3 = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255  # 白底
        # 画预测值（红色）
        for j in range(124):
            if j in cfg.DATASET_COFW.LANDMARK_INDEX:
                landmark = pred_landmarks[j]
                x, y = landmark[0] * 2, landmark[1] * 2
                cv2.circle(image_with_landmarks3, (int(x), int(y)), radius=4, color=colors[indexes[j]-1], thickness=-1)
        # 保存图像
        filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        save_path = os.path.join(save_dir, f'test_{filename}_3.jpg')
        cv2.imwrite(save_path, image_with_landmarks3)

        image_with_landmarks4 = np.ones((h * 2, w * 2, 3), dtype=np.uint8) * 255  # 白底
        # 画预测值（红色）
        for j in range(124):
            if j in cfg.DATASET_AFLW.LANDMARK_INDEX:
                landmark = pred_landmarks[j]
                x, y = landmark[0] * 2, landmark[1] * 2
                cv2.circle(image_with_landmarks4, (int(x), int(y)), radius=4, color=colors[indexes[j]-1], thickness=-1)
        # 保存图像
        filename = os.path.splitext(os.path.basename(filenames[i]))[0]
        save_path = os.path.join(save_dir, f'test_{filename}_4.jpg')
        cv2.imwrite(save_path, image_with_landmarks4)


match_parts_68 = np.array([[0, 16], [1, 15], [2, 14], [3, 13], [4, 12], [5, 11], [6, 10], [7, 9], # outline
            [17, 26], [18, 25], [19, 24], [20, 23], [21, 22], # eyebrow
            [36, 45], [37, 44], [38, 43], [39, 42], [40, 47], [41, 46], # eye
            [31, 35], [32, 34], # nose
            [48, 54], [49, 53], [50, 52], [59, 55], [58, 56], # outer mouth
            [60, 64], [61, 63], [67, 65]])
match_parts_98 = np.array([[0, 32],[1, 31],[2, 30],[3,29],[4, 28],[5, 27],[6, 26],[7, 25],[8, 24],[9, 23],[10, 22],[11, 21],
                           [12, 20],[13, 19],[14, 18],[15, 17], # outline
                           [33, 46],[34, 45],[35, 44],[36, 43],[37, 42],[41, 47],[40, 48],[39, 49],[38, 50],# eyebrow
                           [60, 72],[61, 71],[62, 70],[63, 69],[64, 68],[67, 73],[66, 74],[65, 75], [96, 97], # eye
                           [55, 59],[56, 58], # nose
                           [76, 82],[77, 81],[78, 80],[87, 83],[86, 84], #outer mouth
                           [88, 92],[89, 91],[95, 93]])
match_parts_29 = np.array([[0, 1], [4, 6], [2, 3], [5, 7],  # eyebrow
                  [8, 9], [10, 11], [12, 14], [16, 17], [13, 15],  # eye
                  [18, 19],  # nose
                  [22, 23]])  # mouth
match_parts_19 = np.array([[0, 5], [1, 4], [2, 3],
                  [6, 11], [7, 10], [8, 9],
                  [12, 14],
                  [15, 17]])


def get_preds(scores):  # get_preds的返回值类型： torch.Size([12, 68, 2])
    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    # 输入为4维张量
    assert scores.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(scores.view(scores.size(0), scores.size(1), -1), 2)

    maxval = maxval.view(scores.size(0), scores.size(1), 1)
    idx = idx.view(scores.size(0), scores.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % scores.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / scores.size(3))

    return preds


def compute_nme_ip(preds, targets):
    preds=preds.cpu()
    if isinstance(preds,torch.Tensor):
        preds = preds.numpy()
    target = targets.cpu().numpy()

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = np.linalg.norm(pts_gt[7, ] - pts_gt[10, ])
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[16, ] - pts_gt[17, ])
        elif L == 68:  # 300w
            # interocular
            lcenter = (pts_gt[36,:]+pts_gt[37,:]+pts_gt[38,:]+pts_gt[39,:]+pts_gt[40,:]+pts_gt[41,:])/6
            rcenter = (pts_gt[42,:]+pts_gt[43,:]+pts_gt[44,:]+pts_gt[45,:]+pts_gt[46,:]+pts_gt[47,:])/6
            interpupil = np.linalg.norm(lcenter-rcenter)
            rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interpupil * L)
            continue
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[96, ] - pts_gt[97, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)

    return rmse


# temp_sum = np.zeros(68)


def compute_nme_io(preds, targets, meta):
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    target = targets.cpu().numpy()
    # global temp_sum
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular * L)
        # temp_sum += np.linalg.norm(pts_pred - pts_gt, axis=1)/interocular
    return rmse


# 将最后一个维度翻转
def flip_channels(maps):
    # horizontally flip the channels
    # maps is a tensor of dimension n x c x h x w or c x h x w
    if maps.ndimension() == 4:
        maps = maps.numpy()
        maps = maps[:, :, :, ::-1].copy()
    elif maps.ndimension() == 3:
        maps = maps.numpy()
        maps = maps[:, :, ::-1].copy()
    else:
        exit('tensor dimension is not right')

    return torch.from_numpy(maps).float()


def shuffle_channels_for_horizontal_flipping(maps):
    # when the image is horizontally flipped, its corresponding groundtruth maps should be shuffled.
    # maps is a tensor of dimension n x c x h x w or c x h x w
    if maps.ndimension() == 4:
        dim = 1
        nPoints = maps.size(1)
    elif maps.ndimension() == 3:
        dim = 0
        nPoints = maps.size(0)
    else:
        exit('tensor dimension is not right')
    if nPoints == 98:
        match_parts = match_parts_98
    elif nPoints == 68:
        match_parts = match_parts_68
    elif nPoints == 19:
        match_parts = match_parts_19
    elif nPoints == 29:
        match_parts = match_parts_29
    else:
        exit('points number is not right')
    for i in range(0, match_parts.shape[0]):
        idx1, idx2 = match_parts[i]
        idx1 = int(idx1)
        idx2 = int(idx2)
        tmp = maps.narrow(dim, idx1, 1).clone()  # narrow(dimension, start, length) dimension是要压缩的维度
        maps.narrow(dim, idx1, 1).copy_(maps.narrow(dim, idx2, 1))
        maps.narrow(dim, idx2, 1).copy_(tmp)
    return maps


def validate(config, val_loader, model):
    model.eval()

    nme_batch_sum_ip = 0  # 瞳孔
    nme_batch_sum_io = 0  # 外眼角
    nme_count = 0
    count_failure_010 = 0

    # 用来累积每张图的结果
    # all_filenames = []
    # all_nme_ip = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            input, targets, target_weight, meta = batch
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input = input.to(device)

            num_joints = targets.shape[1]
            if num_joints == 29:
                landmark_index = config.DATASET_COFW.LANDMARK_INDEX
            elif num_joints == 68:
                landmark_index = config.DATASET_300W.LANDMARK_INDEX
            elif num_joints == 98:
                landmark_index = config.DATASET_WFLW.LANDMARK_INDEX
            elif num_joints == 19:
                landmark_index = config.DATASET_AFLW.LANDMARK_INDEX
            else:
                raise TypeError(f"No match for points {num_joints}")

            # 模型推断
            out_heatmap1 = model.encoder.forward_dummy(input)
            # get_points(config, images=input, predict=out_heatmap1,filenames=meta['img_pth'], save_dir="./output_images")
            out_heatmap1 = out_heatmap1[:, landmark_index, :, :]

            images_flip = torch.from_numpy(input.cpu().numpy()[:, :, :, ::-1].copy())  # 左右翻转
            images_flip = images_flip.to(device)

            out_heatmap2 = model.encoder.forward_dummy(images_flip)
            out_heatmap2 = out_heatmap2[:, landmark_index, :, :]
            out_heatmap2 = flip_channels(out_heatmap2.cpu())
            out_heatmap2 = shuffle_channels_for_horizontal_flipping(out_heatmap2)
            out_heatmap = (out_heatmap1.cpu() + out_heatmap2) / 2

            pred_coords = get_preds(out_heatmap)  # 将热图转化为预测的点的坐标

            nme_count = nme_count + targets.shape[0]
            # visualize_with_landmarks(input, pred_coords, targets)

            # filename = os.path.splitext(os.path.basename(meta['img_pth']))[0]

            pred_coords = get_transformer_coords(pred_coords, meta, torch.tensor([meta['output_size'][0][0], meta['output_size'][1][0]]))
            targets = get_transformer_coords(targets, meta, torch.tensor([meta['output_size'][0][0], meta['output_size'][1][0]]))
            # 计算关键点坐标误差 瞳孔归一化
            nme_temp_ip = compute_nme_ip(pred_coords, targets)
            nme_batch_sum_ip += np.sum(nme_temp_ip)
            # 外眼角归一化
            nme_temp_io = compute_nme_io(pred_coords, targets, meta)
            nme_batch_sum_io += np.sum(nme_temp_io)

            # save_landmarks_with_nme(input, pred_coords, targets, meta['img_pth'], "./output_images_wflw/wflw_full", nme_temp_io)
            failure_010 = (nme_temp_ip > 0.10).sum()
            count_failure_010 += failure_010
            # for fname, nme_val in zip(meta['img_pth'], nme_temp_ip):
            #     all_filenames.append(os.path.basename(fname))
            #     all_nme_ip.append(float(nme_val))

        # insert_lists_to_excel("./wflw.xls", all_filenames, all_nme_ip, 3, 1)
        # global temp_sum
        # temp_sum /= nme_count
        # temp_sum.tolist()
        # index = [1, 1, 1, 1, 1, 1, 1, 1, 4, 1,
        #          1, 1, 1, 1, 1, 1, 1, 4, 2, 4,
        #          2, 4, 4, 2, 4, 2, 4, 2, 2, 2,
        #          4, 2, 2, 3, 2, 2, 4, 1, 1, 4,
        #          1, 1, 4, 1, 1, 4, 1, 1, 4, 2,
        #          2, 3, 2, 2, 4, 2, 2, 3, 2, 2,
        #          2, 2, 3, 2, 2, 2, 4, 2]
        # sum_part = [0, 0, 0, 0]
        # for idx, num in enumerate(index):
        #     sum_part[num - 1] += temp_sum[idx]
        #     print(str(idx)+":"+str(temp_sum[idx])+" ")
        # num = [24, 25, 4, 15]
        # for i in range(4):
        #     sum_part[i] /= num[i]
        # print(sum_part)
        failure_010_rate = count_failure_010 / nme_count
        nme_ip = nme_batch_sum_ip / nme_count
        nme_io = nme_batch_sum_io / nme_count
    return nme_ip, nme_io, failure_010_rate


