import re
import numpy as np
import matplotlib.pyplot as plt

# 存储数据的列表
epochs = []
io_300w = []
io_aflw = []
io_wflw = []
io_cofw = []

# 从日志文件中提取数据，假设文件名为'log.txt'
# "G:\uw.log"
with open(r"D:\桌面\BABT\sigma消融\20\log_2025-04-25_20-24-17.log", 'r') as f:
    for line in f:
        # 使用正则表达式提取当前的epoch以及四个指标的值
        epoch_match = re.search(r"Epoch:\s*\[(\d+)", line)
        io300w_match = re.search(r"nme_io_300w_ema=([\d\.]+)", line)
        ioaflw_match = re.search(r"nme_io_aflw=([\d\.]+)", line)
        iowflw_match = re.search(r"nme_io_wflw=([\d\.]+)", line)
        iocofw_match = re.search(r"nme_ip_cofw=([\d\.]+)", line)

        # 仅当行中同时包含四个指标时才提取数据
        if epoch_match and io300w_match and ioaflw_match and iowflw_match and iocofw_match:
            epochs.append(int(epoch_match.group(1)))
            io_300w.append(float(io300w_match.group(1)))
            io_aflw.append(float(ioaflw_match.group(1)))
            io_wflw.append(float(iowflw_match.group(1)))
            io_cofw.append(float(iocofw_match.group(1)))

# 转换为 NumPy 数组方便后续处理
epochs = np.array(epochs)
io_300w = np.array(io_300w)
io_aflw = np.array(io_aflw)
io_wflw = np.array(io_wflw)
io_cofw = np.array(io_cofw)

# 将各指标放入字典中，便于后续统一处理和指定颜色
metrics = {
    "nme_io_300w": {"data": io_300w, "color": "blue"},
    "nme_io_aflw": {"data": io_aflw, "color": "orange"},
    "nme_io_wflw": {"data": io_wflw, "color": "green"},
    "nme_ip_cofw": {"data": io_cofw, "color": "red"},
}

# 分别找到每个指标效果最好的 epoch（假设数值越低越好）
best_epochs = {}
for name, info in metrics.items():
    data = info["data"]
    best_idx = np.argmin(data)  # 效果最好对应的索引
    best_epoch = epochs[best_idx]
    best_epochs[name] = best_epoch

# 获取所有指标的最佳 epoch 并去重
unique_best_epochs = sorted(set(best_epochs.values()))

# 绘制各指标的变化曲线
plt.figure(figsize=(12, 8))
for name, info in metrics.items():
    plt.plot(epochs, info["data"], label=name, marker='', color=info["color"])

# 对每个唯一的最佳 epoch，画竖线，并在竖线与各曲线交点处标注数值
for epoch_val in unique_best_epochs:
    # 画竖线
    plt.axvline(x=epoch_val, linestyle='--', color='gray', alpha=0.5)
    # 找出该 epoch 在数据中的索引（假设每个 epoch 值在数据中只出现一次）
    idx = np.where(epochs == epoch_val)[0][0]
    # 遍历各指标，在交点处标记数值
    for name, info in metrics.items():
        y_val = info["data"][idx]
        # 用 scatter 标出交点
        plt.scatter(epoch_val, y_val, color=info["color"], s=50, zorder=5)
        # 用 annotate 在交点附近标出该点的数值
        plt.annotate(f"{y_val:.4f}", (epoch_val, y_val),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=9, color=info["color"])

plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.legend()
plt.grid(True)
plt.show()
