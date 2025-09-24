import torch
import torch.nn as nn
x1 = torch.randn(1,3,2,2)
x2 = torch.randn(1,3,2,2)

avg = nn.AdaptiveAvgPool2d(1)
conv = nn.Conv2d(3, 1, 1, 1, 0)

n, c, h, w = x1.shape
a = conv(avg(x1)).sigmoid()

b = x1 * a
print(a)
print(a.size())
print(x1)
print(b.size())
print(b)