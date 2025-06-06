import matplotlib.pyplot as plt
import numpy as np
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection


def radar_factory(num_vars, frame='polygon'):
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            verts = unit_poly_verts(theta)
            return plt.Polygon(verts, closed=True, edgecolor='none')

        def _gen_axes_spines(self):
            return {}

    register_projection(RadarAxes)
    return theta


def unit_poly_verts(theta):
    x0, y0, r = [0.5]*3
    return [(r*np.cos(t)+x0, r*np.sin(t)+y0) for t in theta]


# ------------------------- 数据部分 ----------------------------
# labels = [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
labels = ['300W-\nfull', '300W-\nchall', '300W-\ncommon', 'WFLW-\nfull', 'WFLW-\npose', 'WFLW-\nexpression',
          'WFLW-\nillumination', 'WFLW-\nmakeup', 'WFLW-\nocclution', 'WFLW-\nblur', 'AFLW', 'COFW']
data = {
    # 300W(300W-chall 300W-common)
    # WFLW (WFLW-pose WFLW-expression WFLW-illumination WFLW-makeup WFLW-occlution WFLW-blur)
    # AFLW COFW
    'FGTBT(OURS)':       [3.06, 4.51, 2.71, 4.42, 7.43, 4.54, 4.47, 4.37, 5.48, 5.02, 1.67, 4.79],
    'LAB(CVPR18)':        [3.49, 5.19, 2.98, 5.27, 10.24, 5.51, 5.23, 5.15, 6.79, 6.32, 1.85, None],
    'Wing(CVPR18)':       [3.38, 5.23, 2.93, 4.99, 8.75, 5.36, 4.93, 5.41, 6.37, 5.81, None, 5.44],
    'ODN(CVPR19)':        [4.17, 6.67, 3.56, None, None, None, None, None, None, None, None, 5.30],
    'LUVLi(CVPR20)':      [3.23, 5.16, 2.76, None, None, None, None, None, None, None, None, None],
    'SAAT(ICCV21)':       [3.25, 5.03, 2.82, None, None, None, None, None, None, None, None, None],
    # 'DeCaFA':     [3.39, 5.26, 2.93, 4.62, 8.11, 4.65, 4.41, 4.63, 5.74, 5.38, None, None],
    'SLPT(CVPR22)':       [3.17, 4.90, 2.75, None, None, None, None, None, None, None, None, 4.79],
    'PicassoNet(TNNLS23)': [3.58, 5.81, 3.03, 4.82, 8.61, 5.14, 4.73, 4.68, 5.91, 5.56, None, None],
    'GFL(CVPR24)':        [3.20, 4.91, 2.79, None, None, None, None, None, None, None, None, None],
# Liang =      [5.28, 5.45, 3.10, 2.68, 4.86, 1.86]
    # 'GlomFace':   [3.13, 4.79, 2.72, 4.81, 8.71, None, None, None, 5.14, None, 4.37, None]
    # 'BABT':       [4.51, 2.71, 4.42, 1.67, 4.79],
    # 'LAB':        [5.19, 2.98, 5.27, 1.85, None],
    # 'PicassoNet': [5.81, 3.03, 4.82, None, None],
    # 'Wing':       [5.23, 2.93, 4.99, None, 5.44],
    # 'ODN':        [6.67, 3.56, None, None, 5.30]
}

tasks = list(data.keys())
raw_matrix = np.array([
    [np.nan if v is None else v for v in data[task]]
    for task in tasks
], dtype=float)

# 反向归一化并填充缺失
min_per_label = np.nanmin(raw_matrix, axis=0)
normalized = min_per_label / raw_matrix
normalized = np.nan_to_num(normalized, nan=0.0)

# 对比度增强：幂运算
p = 4
contrast = normalized ** p

# ------------------------- 绘图部分 ----------------------------
N = len(labels)
theta = radar_factory(N)
max_radius = contrast.max() * 1.1  # 预留10%空白

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
colors = plt.cm.tab10.colors

for i, task in enumerate(tasks):
    vals = contrast[i].tolist()
    vals += vals[:1]
    thetas = np.concatenate([theta, [theta[0]]])

    ax.plot(thetas, vals, label=task, color=colors[i], linewidth=2)
    ax.fill(thetas, vals, color=colors[i], alpha=0.1)

    # 添加原始值标签
    for t, cval, orig in zip(theta, contrast[i], raw_matrix[i]):
        if not np.isnan(orig):
            ax.text(t, cval + max_radius * 0.02,
                    f"{orig:.2f}",
                    ha='center', va='center',
                    fontsize=9, color=colors[i])

# 设置极轴范围、标签和图例
ax.set_ylim(0, max_radius)
ax.set_varlabels(labels)
ax.set_yticklabels([])
ax.legend(loc='lower right', bbox_to_anchor=(1.3, 1.1))

plt.tight_layout()
plt.show()
