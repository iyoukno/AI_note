'''
@Project ：test 
@File    ：lstm.py
@Author  ：yuk
@Date    ：2024/1/8 16:36 
description：
'''
# import torch.nn as nn
# import torch
# rnn = nn.LSTM(10, 20, 2)
# input = torch.randn(5, 3, 10)
# h0 = torch.randn(2, 3, 20)
# c0 = torch.randn(2, 3, 20)
# output, (hn, cn) = rnn(input, (h0, c0))
# print(output)

# import torch
# import torch.nn as nn
#
# # 定义一个带有Sigmoid激活函数的全连接层
# sigmoid_layer = nn.Sequential(
#     nn.Linear(10, 20),
#     nn.Sigmoid()
# )
#
# input = torch.randn(10)
# out = sigmoid_layer(input)
# print(out)
iterable = [1,2,3,4,5]
include = [1,3]
new_list = [x in include for x in iterable]
print(new_list)