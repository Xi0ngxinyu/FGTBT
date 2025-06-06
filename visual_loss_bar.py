import matplotlib.pyplot as plt
import numpy as np

# 300W
# model1_losses = [0.06299723262117556, 0.037188917685476046, 0.02796630926640183, 0.04141910150038592]    # 0
# model2_losses = [0.06136681679038176, 0.036954616771748806, 0.027681694292539326, 0.04184170552954158]  # 0.3
# model3_losses = [0.06028252752037883, 0.03698132736085865, 0.02804262150605317, 0.0413528751783008]    # 0.6
# model4_losses = [0.06009440411766686, 0.0366270964912013, 0.027601078315573034, 0.0417656415308107]    # 0.9
# model5_losses = [0.060625398436425705, 0.036850726061792516, 0.02890540608865167, 0.04301402611444838]    # 0.999

# WFLW
model1_losses = [0.05442815368514902, 0.03865552870370982, 0.03435261194973265, 0.04324612176103893]    # 0
model2_losses = [0.05283814549710445, 0.03832677404402495, 0.03435336070235579, 0.04293988997979672]  # 0.3
model3_losses = [0.0519845163323567, 0.0380078098148749, 0.03401894341522731, 0.04261716383454475]    # 0.6
model4_losses = [0.05127935832100134, 0.03810632477328954, 0.034301740064820854, 0.042901115832501176]    # 0.9
model5_losses = [0.052183460243511934, 0.038518153595011075, 0.03468548856551448, 0.04362206551562995]   # 0.999

# x 轴的位置（part 编号）
x = np.arange(1, 5)

# 调整柱子的宽度（这里设为 0.12，让五根柱子并列时间隔合适）
bar_width = 0.12

plt.figure(figsize=(6, 4))
plt.ylim(0.03, 0.055)  # 根据你的数据区间调整 y 轴范围

# 依次偏移画出五个模型
plt.bar(x - 2*bar_width, model1_losses, width=bar_width, label='0', color='#46788e')
plt.bar(x -     bar_width, model2_losses, width=bar_width, label='0.3', color='#78b7c9')
plt.bar(x             , model3_losses, width=bar_width, label='0.6', color='#f6e093')
plt.bar(x +     bar_width, model4_losses, width=bar_width, label='0.9', color='#e58b7b')
plt.bar(x + 2*bar_width, model5_losses, width=bar_width, label='0.999', color='#97b319')

# 如果需要在每个柱子上方标出数值，可以参考下面的写法：
# for i in range(4):
#     # Model 1
#     height = model1_losses[i]
#     plt.text(x[i] - 2*bar_width, height + 0.002, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
#     # Model 2
#     height = model2_losses[i]
#     plt.text(x[i] -     bar_width, height + 0.002, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
#     # Model 3
#     height = model3_losses[i]
#     plt.text(x[i]             , height + 0.002, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
#     # Model 4
#     height = model4_losses[i]
#     plt.text(x[i] +     bar_width, height + 0.002, f'{height:.3f}', ha='center', va='bottom', fontsize=8)
#     # Model 5
#     height = model5_losses[i]
#     plt.text(x[i] + 2*bar_width, height + 0.002, f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# 添加标签、图例等
plt.xlabel('Frequency')
plt.ylabel('Avg Loss')
plt.xticks(x, ['1', '2', '3', '4'])
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
