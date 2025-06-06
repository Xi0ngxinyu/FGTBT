import numpy as np
import matplotlib.pyplot as plt

# 定义 T
T = 200000

# 定义 t 的取值范围
t = np.linspace(1, T, 1000)  # 取 1000 个点以保证曲线平滑

# 定义 a 的取值范围
a_values = [1, 2, 3, 4]

# 绘制图像
plt.figure(figsize=(8, 6))
for a in a_values:
    y = (1 - (t/T)) / (1 - (t/T)**a)
    plt.plot(t, y, label=f'a = {a}')

# 设置图例和标签
plt.xlabel('t')
plt.ylabel('Function Value')
plt.title('Plot of (1-(t/T)) / (1-(t/T)^a) for different a values')
plt.legend()
plt.grid()

# 显示图像
plt.show()
