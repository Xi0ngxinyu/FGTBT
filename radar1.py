import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = 'sans-serif'

# 数据准备\n",
labels = ['WFLW'+'\n'+'(NME_io)',
        'COFW'+'\n'+'(NME_ip)',
        '300W-Full'+'\n'+'(NME_io)',
        '300W-Comm'+'\n'+'(NME_io)',
        '300W-Chal'+'\n'+'(NME_io)',
        'AFLW-full'+'\n'+'(NME_box)',
]

num_vars = len(labels)

# 定义每个方向的数值和范围
# AirNet =   [30.91, 21.04, 32.98, 24.35, 18.18]
# DGUNet =   [31.10, 24.78, 36.62, 27.25, 21.87, 28.32]

# LAB{$\rm _{CVPR18}$}\cite{Wu2018LookAB}  &5.27 &7.56 &0.532 &- &3.49 &2.98 &5.19 &1.85  \\
# Wing{$\rm _{CVPR18}$}\cite{Feng2017WingLF} &4.99 &6.00 &0.550 &5.44 &- &- &- &1.47   \\
# DeCaFa{$\rm _{ICCV19}$}\cite{Dapogny2019DeCaFADC} &4.62 &4.84 &0.563 &- &3.39 &2.93 &5.26 &-   \\
# HRNet{$\rm _{19'}$}\cite{Sun2019HighResolutionRF} &4.6 &4.64 &- &- &3.32 &2.87 &5.15 &1.57   \\
# Awing{$\rm _{ICCV19}$}\cite{Wang2019AdaptiveWL} &4.36 &2.84 &0.572 &4.94 &3.07 &2.72 &4.52 &1.53   \\
# AVS + SAN{$\rm _{ICCV19}$}\cite{Qian2019AggregationVS} &4.39 &4.08 &0.591 &- &3.86 &3.21 &6.46 &-   \\
# DAG{$\rm _{ECCV20}$}\cite{Li2020StructuredLD} &4.21 &3.04 &0.589 &- &3.04 &2.62 &4.77 &-   \\
# LUVLi{$\rm _{CVPR20}$}\cite{Kumar2020LUVLiFA} &4.37 &3.12 &0.577 &- &3.23 &2.76 &5.16 &1.39  \\
# ADNet{$\rm _{ICCV21}$}\cite{Huang2021ADNetLE} &4.14 &2.72 &0.602 &4.68 &2.93 &2.53 &4.58 &- \\
# PIPNet{$\rm _{IJCV21}$}\cite{Jin2020PixelinPixelNT} &4.31 &- &- &- &3.19 &2.78 &4.89 &1.48   \\
# SLPT{$\rm _{CVPR22}$}\cite{Xia2022SparseLP} &4.14 &2.76 &0.595 &4.79 &3.17 &2.75 &4.90 &-   \\
# DTLD{$\rm _{CVPR22}$}\cite{Li2022TowardsAF} &4.08 &2.76 &- &- &2.96 &2.59 &4.50 &1.37   \\
# STAR{$\rm _{CVPR23}$}\cite{zhou2023star}  &4.02 &2.32 &0.605 &4.62 &2.87 &2.52 &4.32 \\
# Liang et al. {$\rm _{CVPR24}$}\cite{liang2024generalizable} &- &- &- &- &3.10 &2.68 &4.86 &-\\

LAB =        [5.27, 5.45, 3.49, 2.98, 5.19, 1.85]
Decafa =     [4.62, 5.44, 3.39, 2.93, 5.26, 1.86]
HRNet =      [4.6,  5.45, 3.32, 2.87, 5.15, 1.57]
Awing =      [4.36, 4.94, 3.07, 2.72, 4.52, 1.53]
AVS_SAN =    [4.39, 5.45, 3.86, 3.21, 6.46, 1.86]
DAG =        [4.21, 5.45, 3.04, 2.62, 4.77, 1.86]
LUVLi =      [4.37, 5.45, 3.23, 2.76, 5.16, 1.39]
ADNet =      [4.14, 4.68, 2.93, 2.53, 4.58, 1.86]
SLPT =       [4.14, 4.79, 3.17, 2.75, 4.90, 1.86]
DTLD =       [4.08, 5.45, 2.96, 2.59, 4.50, 1.37]
STAR =       [4.02, 4.62, 2.87, 2.52, 4.32, 1.86]
Liang =      [5.28, 5.45, 3.10, 2.68, 4.86, 1.86]
FreeAlign =  [4.02, 4.51, 2.84, 2.47, 4.35, 1.31]
FreeAlign1 =  [3.99, 4.46, 2.84, 2.47, 4.35, 1.28]


# ranges = [(4.6, 5.27), (4.64, 7.56), (0.001, 0.563), (0.001, 0.003), (3.32, 3.49), (2.87, 2.98), (5.15, 5.26), (0.001, 1.57)]

ranges = [(3.99, 5.28), (4.46, 5.45), (2.84, 3.86), (2.47, 3.21), (4.32, 6.46), (1.28, 1.86)]

# 计算角度\n",
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

# 将值缩放到 [0, 1] 范围
def scale_values(values, ranges):
    return [1-(values[i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0]) for i in range(len(ranges))]

# return [(values[i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0]) for i in range(len(ranges))]


def inv_scale_value(values, ranges):
    return [1 - (values - ranges[0]) / (ranges[1] - ranges[0])]

def inv_scale_values(values, ranges):
    return [1 - (values[i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0]) for i in range(len(ranges))]

# 注意这里得多加个值\n"
LAB_values = inv_scale_values(LAB, ranges) + inv_scale_values(LAB, ranges)[:1]
Decafa_values = inv_scale_values(Decafa, ranges) + inv_scale_values(Decafa, ranges)[:1]
HRNet_values = inv_scale_values(HRNet, ranges) + inv_scale_values(HRNet, ranges)[:1]
Awing_values = inv_scale_values(Awing, ranges) + inv_scale_values(Awing, ranges)[:1]
AVS_SAN_values = inv_scale_values(AVS_SAN, ranges) + inv_scale_values(AVS_SAN, ranges)[:1]
DAG_values = inv_scale_values(DAG, ranges) + inv_scale_values(DAG, ranges)[:1]
LUVLi_values = inv_scale_values(LUVLi, ranges) + inv_scale_values(LUVLi, ranges)[:1]
ADNet_values = inv_scale_values(ADNet, ranges) + inv_scale_values(ADNet, ranges)[:1]
SLPT_values = inv_scale_values(SLPT, ranges) + inv_scale_values(SLPT, ranges)[:1]
DTLD_values = inv_scale_values(DTLD, ranges) + inv_scale_values(DTLD, ranges)[:1]
STAR_values = inv_scale_values(STAR, ranges) + inv_scale_values(STAR, ranges)[:1]
Liang_values = inv_scale_values(Liang, ranges) + inv_scale_values(Liang, ranges)[:1]
FreeAlign_values = inv_scale_values(FreeAlign, ranges) + inv_scale_values(FreeAlign, ranges)[:1]
FreeAlign1_values = inv_scale_values(FreeAlign1, ranges) + inv_scale_values(FreeAlign1, ranges)[:1]



# MambaIR_values = scale_values(MambaIR, ranges) + scale_values(MambaIR, ranges)[:1]
#
# IDR_values = scale_values(IDR, ranges) + scale_values(IDR, ranges)[:1]
# PromptIR_values = scale_values(PromptIR, ranges) + scale_values(PromptIR, ranges)[:1]
# Gridformer_values = scale_values(Gridformer, ranges) + scale_values(Gridformer, ranges)[:1]
# InstructIR_values = scale_values(InstructIR, ranges) + scale_values(InstructIR, ranges)[:1]
# PerceiveIR_values = scale_values(PerceiveIR, ranges) + scale_values(PerceiveIR, ranges)[:1]

# 绘图\n",
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# 自定义雷达图样式\n",
# ax.set_facecolor("\"#f2f2f2\")

ax.spines['polar'].set_linestyle('--')
ax.spines['polar'].set_linewidth(0.5)
ax.spines['polar'].set_color('gray')# 设置圆形边框的颜色\n",
ax.spines['polar'].set_visible(False)  # 隐藏圆形边框, True是不隐藏\n",
ax.grid(color='gray', linestyle='-', linewidth=0.5)  # 设置网格线样式\n",
ax.tick_params(colors='gray', labelsize=12)  # 设置刻度样式并增加label与刻度之间的距离 pad可设距离

# 绘制雷达图线条和填充区域\n",
ax.fill(angles, LAB_values, color='darkviolet', alpha=0.1, label='LAB')  # 0.15
ax.plot(angles, LAB_values, color='darkviolet', linewidth=1.3)
ax.fill(angles, Decafa_values, color='green', alpha=0.1, label='Decafa')
ax.plot(angles, Decafa_values, color='green', linewidth=0.8)
ax.fill(angles, HRNet_values, color='darkorange', alpha=0.1, label='HRNet')
ax.plot(angles, HRNet_values, color='darkorange', linewidth=0.8)
ax.fill(angles, Awing_values, color='orange', alpha=0.1, label='Awing')  # 0.15
ax.plot(angles, Awing_values, color='orange', linewidth=1.3)
ax.fill(angles, AVS_SAN_values, color='cyan', alpha=0.1, label='AVS_SAN')
ax.plot(angles, AVS_SAN_values, color='cyan', linewidth=0.8)
ax.fill(angles, DAG_values, color='magenta', alpha=0.1, label='DAG')
ax.plot(angles, DAG_values, color='magenta', linewidth=0.8)
ax.fill(angles, LUVLi_values, color='darkgoldenrod', alpha=0.1, label='LUVLi')  # 0.15
ax.plot(angles, LUVLi_values, color='darkgoldenrod', linewidth=1.3)
ax.fill(angles, ADNet_values, color='blue', alpha=0.1, label='ADNet')
ax.plot(angles, ADNet_values, color='blue', linewidth=0.8)
ax.fill(angles, SLPT_values, color='pink', alpha=0.1, label='SLPT')
ax.plot(angles, SLPT_values, color='pink', linewidth=0.8)
ax.fill(angles, DTLD_values, color='gray', alpha=0.1, label='DTLD')  # 0.15
ax.plot(angles, DTLD_values, color='gray', linewidth=1.3)
ax.fill(angles, STAR_values, color='yellow', alpha=0.1, label='STAR')
ax.plot(angles, STAR_values, color='yellow', linewidth=0.8)
ax.fill(angles, Liang_values, color='royalblue', alpha=0.1, label='Liang')
ax.plot(angles, Liang_values, color='royalblue', linewidth=0.8)
ax.fill(angles, FreeAlign_values, color='red', alpha=0.1, label='FreeAlign')
ax.plot(angles, FreeAlign_values, color='red', linewidth=0.8)
ax.fill(angles, FreeAlign1_values, color='purple', alpha=0.1, label='FreeAlign*')
ax.plot(angles, FreeAlign1_values, color='purple', linewidth=0.8)

# 设置轴标签\n",
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, size=13, color='black')  # SIZE是字体，每个角度的上的标签\n",

# 标出数值\n",
for i in range(0, 6):
    ax.text(angles[i], FreeAlign1_values[i]-0.1, f'{FreeAlign1[i]}', horizontalalignment='center', verticalalignment='center', size=8, color='purple')

# ax.text(angles[4], STAR_values[4]-0.08, f'{STAR[4]}', horizontalalignment='center', verticalalignment='center', size=10, color='yellow')

# 添加原点为 N/A 的标签\n",
# 去除默认的径向标签\n",
ax.set_yticklabels([])
ax.set_yticks(np.array([0.2,0.4,0.6,0.8,1]))
ax.set_ylim(0,1.2)  # 表示最外面圆圈的位置，常规是1；可设置的更大\n",

# 添加标题和图例\n",
# plt.title('带多种图例和不同方向范围的雷达图', size=20, color='blue', y=1.1)
custom_lines = [
    Rectangle((0, 0), 1, 1, fc='darkviolet', alpha=0.8, label='LAB(CVPR18)'),
    Rectangle((0, 0), 1, 1, fc='darkorange', alpha=0.8, label='HRNet(19)'),
    Rectangle((0, 0), 1, 1, fc='cyan', alpha=0.8, label='AVS_SAN(ICCV19)'),
    Rectangle((0, 0), 1, 1, fc='magenta', alpha=0.8, label='DAG(ECCV20)'),
    Rectangle((0, 0), 1, 1, fc='darkgoldenrod', alpha=0.8, label='LUVLi(CVPR20)'),
    Rectangle((0, 0), 1, 1, fc='blue',  alpha=0.8, label='ADNet(ICCV21)'),
    Rectangle((0, 0), 1, 1, fc='pink', alpha=0.8, label='SLPT(CVPR22)'),
    Rectangle((0, 0), 1, 1, fc='gray', alpha=0.8, label='DTLD(CVPR22)'),
    Rectangle((0, 0), 1, 1, fc='green', alpha=0.8, label='Decafa(ICCV19)'),
    Rectangle((0, 0), 1, 1, fc='royalblue',  alpha=0.8, label='Liang(CVPR24)'),
    Rectangle((0, 0), 1, 1, fc='yellow', alpha=0.8, label='STAR(CVPR23)'),
    Rectangle((0, 0), 1, 1, fc='red', alpha=0.8, label='FreeAlign(Ours)'),
    Rectangle((0, 0), 1, 1, fc='purple', alpha=0.8, label='FreeAlign*(Ours)'),
]
# edgecolor='gray', 画的框边缘加颜色\n",
plt.legend(handles=custom_lines, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.25), fontsize=12)
plt.savefig('./radar_new_1.png',bbox_inches='tight', pad_inches = 0.03, dpi=300)
plt.show()