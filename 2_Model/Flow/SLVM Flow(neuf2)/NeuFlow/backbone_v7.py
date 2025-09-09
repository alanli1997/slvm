import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    # C_i_r
    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, act=True):
        super().__init__()
        p = k//2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.InstanceNorm2d(c2)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ConvBlock(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode='zeros', bias=False)

        self.conv2 = torch.nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = torch.nn.LeakyReLU(negative_slope=0.1, inplace=False)

        self.norm1 = torch.nn.BatchNorm2d(out_planes)

        self.norm2 = torch.nn.BatchNorm2d(out_planes)

        # self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):

        # x = self.dropout(x)

        x = self.relu(self.norm1(self.conv1(x)))
        x = self.relu(self.norm2(self.conv2(x)))
        # x = self.relu(self.conv1(x))
        # x = self.relu(self.conv2(x))

        return x

    def forward_fuse(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        return x

# Equal to the LAPM in paper
# ----------------------
class DAF1(nn.Module):
    def __init__(self, c1=3, c2=1, th=0.02, stride=1):
        super().__init__()
        self.th = th
        self.ds = nn.AvgPool2d(2, 2) if stride==2 else nn.Identity()
        # self.pfactor = 10
        # self.pfactor_param = nn.Parameter(torch.tensor(6.906), requires_grad=True) # 6.906=10
        self.cv = Conv(1, c2, 1, 1)

    def forward(self, x):
        # x = (0.1 + torch.sigmoid(self.pfactor_param)) * 10 * self.ds(x)
        x = 10 * self.ds(x)
        # gray_tensor = (0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :])
        # binary_tensor = torch.where(gray_tensor > self.th, x.new_ones(gray_tensor.size()),
        #                             x.new_zeros(gray_tensor.size())).unsqueeze(0)
        weights = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        gray_tensor = torch.sum(x * weights, dim=1, keepdim=True)
        binary_tensor = torch.where(gray_tensor > self.th, x.new_ones(gray_tensor.size()),
                                    x.new_zeros(gray_tensor.size()))
        return self.cv(binary_tensor)


class DAF2(nn.Module):
    def __init__(self, c1=1, c2=1, th=0.02, stride=2):
        super().__init__()
        self.th = th
        self.ds = nn.AvgPool2d(2, 2) if stride == 2 else nn.Identity()
        # self.cv = nn.Conv2d(2*c1, c2, 1, 1, bias=False)
        self.cv = Conv(2 * c1, c2, 1, 1)

    def forward(self, x):
        gray_tensor = self.ds(x)
        binary_tensor = torch.where(gray_tensor > self.th,
                                    x.new_ones(gray_tensor.size()),
                                    x.new_zeros(gray_tensor.size()))
        # binary_tensor = torch.where(gray_tensor > self.th, 1.0, 0.0)
        return self.cv(torch.cat((binary_tensor, gray_tensor), 1))
# --------------------


class Fuse(nn.Module):
    def __init__(self, c1, c2):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        self.cv = Conv(c1, c2, 1, 1)

    def forward(self, x1, x2):
        return self.cv(torch.cat((x1, x2), 1))


class SLVM(torch.nn.Module):
    def __init__(self, feature_dim_s16, context_dim_s16, feature_dim_s8, context_dim_s8):
        super(SLVM, self).__init__()

        self.daf2 = DAF1(stride=2)
        self.daf4 = DAF2(stride=2)
        self.daf8 = DAF2(stride=2)

        self.f2 = Fuse(4, 3)
        self.f4 = Fuse(4, 3)
        self.f8 = Fuse(4, 3)

        self.block_8_1 = ConvBlock(3, feature_dim_s8 * 2, kernel_size=8, stride=4, padding=2)

        self.block_8_2 = ConvBlock(3, feature_dim_s8, kernel_size=6, stride=2, padding=2)

        self.block_cat_8 = ConvBlock(feature_dim_s8 * 3, feature_dim_s8 + context_dim_s8, kernel_size=3, stride=1, padding=1)

        self.block_16_1 = ConvBlock(3, feature_dim_s16, kernel_size=6, stride=2, padding=2)

        self.block_8_16 = ConvBlock(feature_dim_s8 + context_dim_s8, feature_dim_s16, kernel_size=6, stride=2, padding=2)

        self.block_cat_16 = ConvBlock(feature_dim_s16 * 2, feature_dim_s16 + context_dim_s16 - 2, kernel_size=3, stride=1, padding=1)

    def init_pos(self, batch_size, height, width, device, amp):
        ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.half if amp else torch.float, device=device),
                                torch.arange(width, dtype=torch.half if amp else torch.float, device=device), indexing='ij')
        ys = (ys-height/2)
        xs = (xs-width/2)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.pos_s16 = self.init_pos(batch_size, height, width, device, amp)

    def forward(self, img):


        # dx2 = self.daf2(dx1)
        # dx3 = self.daf3(dx2)
        # x = self.f1(dx1, x)
        # x = self.f2(dx2, x)
        # x = self.f3(dx3, x)

        dx = self.daf2(img)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)  # 1/2
        img = self.f2(dx, img)

        x_8 = self.block_8_1(img)  # 1/8

        img = F.avg_pool2d(img, kernel_size=2, stride=2)  # 1/4
        dx = self.daf4(dx)
        img = self.f4(img, dx)

        x_8_2 = self.block_8_2(img)  # 1/8

        x_8 = self.block_cat_8(torch.cat([x_8, x_8_2], dim=1))  # 1/8

        img = F.avg_pool2d(img, kernel_size=2, stride=2)   # 1/8
        dx = self.daf8(dx)
        img = self.f8(dx, img)

        x_16 = self.block_16_1(img)  # 1/16

        x_16_2 = self.block_8_16(x_8)   # 1/16

        x_16 = self.block_cat_16(torch.cat([x_16, x_16_2], dim=1))  # 1/16

        x_16 = torch.cat([x_16, self.pos_s16], dim=1)

        return x_16, x_8


class CNNEncoder(torch.nn.Module):
    def __init__(self, feature_dim_s16, context_dim_s16, feature_dim_s8, context_dim_s8):
        super(CNNEncoder, self).__init__()

        self.daf2 = DAF1(stride=2)
        self.daf4 = DAF2(stride=2)
        self.daf8 = DAF2(stride=2)

        self.f2 = Fuse(4, 3)
        self.f4 = Fuse(4, 3)
        self.f8 = Fuse(4, 3)

        self.block_8_1 = ConvBlock(3, feature_dim_s8 * 2, kernel_size=8, stride=4, padding=2)

        self.block_8_2 = ConvBlock(3, feature_dim_s8, kernel_size=6, stride=2, padding=2)

        self.block_cat_8 = ConvBlock(feature_dim_s8 * 3, feature_dim_s8 + context_dim_s8, kernel_size=3, stride=1, padding=1)

        self.block_16_1 = ConvBlock(3, feature_dim_s16, kernel_size=6, stride=2, padding=2)

        self.block_8_16 = ConvBlock(feature_dim_s8 + context_dim_s8, feature_dim_s16, kernel_size=6, stride=2, padding=2)

        self.block_cat_16 = ConvBlock(feature_dim_s16 * 2, feature_dim_s16 + context_dim_s16 - 2, kernel_size=3, stride=1, padding=1)

    def init_pos(self, batch_size, height, width, device, amp):
        ys, xs = torch.meshgrid(torch.arange(height, dtype=torch.half if amp else torch.float, device=device),
                                torch.arange(width, dtype=torch.half if amp else torch.float, device=device), indexing='ij')
        ys = (ys-height/2)
        xs = (xs-width/2)
        pos = torch.stack([ys, xs])
        return pos[None].repeat(batch_size,1,1,1)

    def init_bhwd(self, batch_size, height, width, device, amp):
        self.pos_s16 = self.init_pos(batch_size, height, width, device, amp)

    def forward(self, img):


        # dx2 = self.daf2(dx1)
        # dx3 = self.daf3(dx2)
        # x = self.f1(dx1, x)
        # x = self.f2(dx2, x)
        # x = self.f3(dx3, x)

        dx = self.daf2(img)
        img = F.avg_pool2d(img, kernel_size=2, stride=2)  # 1/2
        img = self.f2(dx, img)

        x_8 = self.block_8_1(img)  # 1/8

        img = F.avg_pool2d(img, kernel_size=2, stride=2)  # 1/4
        dx = self.daf4(dx)
        img = self.f4(img, dx)

        x_8_2 = self.block_8_2(img)  # 1/8

        x_8 = self.block_cat_8(torch.cat([x_8, x_8_2], dim=1))  # 1/8

        img = F.avg_pool2d(img, kernel_size=2, stride=2)   # 1/8
        dx = self.daf8(dx)
        img = self.f8(dx, img)

        x_16 = self.block_16_1(img)  # 1/16

        x_16_2 = self.block_8_16(x_8)   # 1/16

        x_16 = self.block_cat_16(torch.cat([x_16, x_16_2], dim=1))  # 1/16

        x_16 = torch.cat([x_16, self.pos_s16], dim=1)

        return x_16, x_8