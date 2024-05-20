'''
@Project ：test 
@File    ：caogao.py
@Author  ：yuk
@Date    ：2024/4/7 16:24 
description：
'''


# import torch
#
# # # 创建一个示例的张量，假设为您的输入张量
# # input_tensor = torch.randn(1, 3, 256, 128)
# #
# # # 获取索引为2的位置到最后的切片
# # slice_tensor = input_tensor[:, :, 2:]
# #
# # print("Output tensor shape:", output_tensor.shape)
# #
# # # 使用 torch.cat 函数复制8份并在第二个维度上拼接起来
# # output_tensor = torch.cat([slice_tensor] * 8, dim=2)
# #
# # print("Input tensor shape:", input_tensor.shape)
#
# l = torch.hub.list('pytorch/vision', force_reload=True)
# print(l)

# import shutil
# for i in range(9):
#     i += 1
#     shutil.copyfile(r'D:\datasets\VOC2012\relabel_fire\vlcsnap-2024-05-14-09h12m57s166.jpg', r'D:\datasets\VOC2012\relabel_fire\vlcsnap-2024-05-14-09h12m57s166-'+ str(i) +'.jpg')

# 定义一个装饰器函数，用于记录函数的调用日志
def log(func):
    def wrapper(*args, **kwargs):
        print(f"调用函数 {func.__name__}，参数：{args}, {kwargs}")
        return func(*args, **kwargs)

    return wrapper


# 应用装饰器
@log
def add(x, y, k=2, b=3):
    return x + y


# 调用被装饰的函数
result = add(3, 4, k=2, b=3)
print(result)  # 输出 7
