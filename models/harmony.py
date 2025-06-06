import torch
import torch.nn as nn
import torch.optim as optim


# # 假设你的模型有4个任务
# class MultiTaskModel(nn.Module):
#     def __init__(self):
#         super(MultiTaskModel, self).__init__()
#         # 你的模型定义
#         # 可以是不同任务的子网络，或者一个共享网络
#         self.shared_layer = nn.Linear(256, 256)
#         self.task1_head = nn.Linear(256, 1)
#         self.task2_head = nn.Linear(256, 1)
#         self.task3_head = nn.Linear(256, 1)
#         self.task4_head = nn.Linear(256, 1)
#
#     def forward(self, x):
#         shared = self.shared_layer(x)
#         task1_out = self.task1_head(shared)
#         task2_out = self.task2_head(shared)
#         task3_out = self.task3_head(shared)
#         task4_out = self.task4_head(shared)
#         return task1_out, task2_out, task3_out, task4_out

#
# # 计算任务权重的函数
# def compute_task_weights(L_iter, L_smoothed_task, L_smoothed_points, beta=0.01, tau=3):
#     # 计算并返回更新后的平滑损失、最小损失和任务权重
#     # 使用循环更新平滑损失
#     # for i in range(len(L_iter)):
#     with torch.no_grad():
#         L_smoothed_task = beta * L_smoothed_task + (1 - beta) * L_iter.mean()
#         L_smoothed_points = beta * L_smoothed_points + (1 - beta) * L_iter
#         points_level_coverage_rate = L_iter / L_smoothed_points
#         # task_level_coverage_rate = L_iter.mean() / L_smoothed_task
#         # s_t = points_level_coverage_rate / task_level_coverage_rate  # 收敛速率=当前轮的损失/平滑后的损失
#
#         # L_min = torch.min(L_iter, L_min)  # 所有轮的最小损失
#         # r_t = L_iter / _L_min  # 全局收敛速率=当前轮的损失/全局最小损失
#         exp_values = torch.exp(points_level_coverage_rate / tau)
#         sum_exp_values = torch.sum(exp_values)
#         alpha_t = exp_values / sum_exp_values  # 最后的动态权重结果
#     return L_smoothed_task, L_smoothed_points, alpha_t
def compute_task_weights(L_iter, L_min, L_smoothed, beta=0.01, tau=3):
    # 计算并返回更新后的平滑损失、最小损失和任务权重
    with torch.no_grad():
        L_smoothed = beta * L_smoothed + (1 - beta) * L_iter
        s_t = L_iter /L_smoothed
        L_min = torch.min(L_iter, L_min)
        r_t = L_min/ L_iter
        x = torch.exp(s_t*r_t/ tau)

        mean = x.mean()
        std = x.std()
        epsilon = 1e-8  # 防止除以0的微小常数
        std_adj = std if std > epsilon else epsilon
        x_norm = (x - mean) / std_adj
        target_mean = 1.0
        # target_std = 0.001
        alpha_t = x_norm * std + target_mean
    return L_smoothed, L_min, alpha_t

# 模型初始化
# model = MultiTaskModel()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
#
# # 假设你有4个任务的损失函数
# def task_loss_fn(task_output, target):
#     return torch.mean((task_output - target) ** 2)
#
#
# # 假设你有4个数据集，每个任务的数据
# data_loader_task1 = [(torch.randn(32, 256), torch.randn(32, 1))]  # (input, target) for task 1
# data_loader_task2 = [(torch.randn(32, 256), torch.randn(32, 1))]  # (input, target) for task 2
# data_loader_task3 = [(torch.randn(32, 256), torch.randn(32, 1))]  # (input, target) for task 3
# data_loader_task4 = [(torch.randn(32, 256), torch.randn(32, 1))]  # (input, target) for task 4
#
# # 超参数
# beta = 0.01  # EMA的平滑因子
# tau = 3  # 温度系数
#
# # 初始损失和历史最小损失
# L_prev_iter = torch.zeros(4)  # 初始时，假设之前的损失为0
# L_smoothed = torch.zeros(4)  # 初始平滑损失
# L_min = torch.tensor([float('inf')] * 4)  # 初始历史最小损失为无穷大
#
# # 训练过程
# for epoch in range(100):  # 假设训练100个epoch
#     model.train()
#
#     # 在每次迭代时，获取当前任务的损失并计算任务权重
#     for (data_task1, target_task1), (data_task2, target_task2), (data_task3, target_task3), (
#     data_task4, target_task4) in zip(data_loader_task1, data_loader_task2, data_loader_task3, data_loader_task4):
#         # 前向传播
#         task1_out, task2_out, task3_out, task4_out = model(data_task1)
#
#         # 计算当前任务的损失
#         L_iter_task1 = task_loss_fn(task1_out, target_task1)
#         L_iter_task2 = task_loss_fn(task2_out, target_task2)
#         L_iter_task3 = task_loss_fn(task3_out, target_task3)
#         L_iter_task4 = task_loss_fn(task4_out, target_task4)
#
#         L_iter = torch.stack([L_iter_task1, L_iter_task2, L_iter_task3, L_iter_task4])  # 当前迭代的损失
#
#         # 更新损失相关值
#         L_smoothed, L_min, alpha_t = compute_task_weights(L_iter, L_min, L_smoothed, beta, tau)
#
#         # 计算加权损失
#         weighted_loss = torch.sum(alpha_t * L_iter)  # 每个任务损失加权求和
#
#         # 反向传播
#         optimizer.zero_grad()  # 清零梯度
#         weighted_loss.backward()  # 反向传播
#         optimizer.step()  # 更新模型参数
#
#         # 更新上一轮损失（为了下次计算收敛率）
#         L_prev_iter = L_iter
#
#     print(f"Epoch [{epoch + 1}/100], Weighted Loss: {weighted_loss.item()}")
