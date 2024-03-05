'''
@Project ：test 
@File    ：next_iter.py
@Author  ：yuk
@Date    ：2024/3/4 14:18 
description：
'''
# l = [1,3,4,5,2,7]
# l_iter = iter(l)
# print(next(l_iter))
# print(next(l_iter))
# print(l.__class__.__name__)

from math import log

print(1 * log(0.2))

import torch
import torch.nn as nn
bceloss = nn.BCELoss()

pred = torch.tensor([1], dtype=torch.float)
label = torch.tensor([1], dtype=torch.float)
print(bceloss(pred, label))

