import torch

# 初始化 6x6 矩阵和单元素
matrix = torch.arange(1, 37).view(6, 6)  # 6x6 的矩阵
element = torch.tensor([99])             # 单元素

# 拆分成 4 个 3x3 的矩阵
top_left = matrix[:3, :3]
top_right = matrix[:3, 3:]
bottom_left = matrix[3:, :3]
bottom_right = matrix[3:, 3:]

# 创建 7x7 的全零矩阵
result = torch.zeros(7, 7, dtype=matrix.dtype)

# 填入 3x3 子矩阵
result[:3, :3] = top_left           # 左上
result[:3, 4:] = top_right          # 右上
result[4:, :3] = bottom_left        # 左下
result[4:, 4:] = bottom_right       # 右下

# 插入单元素到正中间
result[3, 3] = element

# 打印结果
print("Original 6x6 Matrix:")
print(matrix)
print("\nResulting 7x7 Matrix:")
print(result)
