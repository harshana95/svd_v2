import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.nn.modules.loss import _Loss


def MSE(**kwargs):
    """
    L2 loss
    """
    return nn.MSELoss()


def L1(**kwargs):
    """
    L1 loss
    """
    return nn.L1Loss()


def Huber(**kwargs):
    return nn.HuberLoss()


class BinaryCrossEntropyLoss(_Loss):
    def __init__(self, **kwargs):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, x, y):
        return nn.functional.binary_cross_entropy(x, y)


class L1GradientLoss(_Loss):
    """
    Gradient loss
    """

    def __init__(self, **kwargs):
        super(L1GradientLoss, self).__init__()
        self.get_grad = Gradient()
        self.L1 = nn.L1Loss()

    def forward(self, x, y):
        grad_x = self.get_grad(x)
        grad_y = self.get_grad(y)
        loss = self.L1(grad_x, grad_y)
        return loss


class Gradient(nn.Module):
    def __init__(self):
        super(Gradient, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()

    def forward(self, x):
        x0 = x[:, 0].cuda()
        # x1 = x[:, 1]
        # x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=2)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=2)
        #
        # x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=2)
        # x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=2)
        #
        # x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=2)
        # x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=2)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        # x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        # x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        # x = torch.cat([x0, x1, x2], dim=1)
        x = torch.cat([x0], dim=1)
        return x


class L1_Charbonnier_loss(_Loss):
    """
    L1 Charbonnierloss
    """

    def __init__(self, **kwargs):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


class L1_Charbonnier_loss_color(_Loss):
    """
    L1 Charbonnierloss color
    """

    def __init__(self, **kwargs):
        super(L1_Charbonnier_loss_color, self).__init__()
        self.eps = 1e-3

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        diff_sq = diff * diff
        # print(diff_sq.shape)
        diff_sq_color = torch.mean(diff_sq, 1, True)
        # print(diff_sq_color.shape)
        error = torch.sqrt(diff_sq_color + self.eps * self.eps)
        loss = torch.mean(error)
        return loss


class Perceptual_loss(_Loss):
    @staticmethod
    def getContentFunc():
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model
    def preprocess(self, img):
        mean_bgr = torch.tensor((103.939, 116.779, 123.68), device=img.device).reshape(1, 3, 1, 1)
        if img.shape[1] == 1:
            # print("Grayscale image for VGG loss?")
            img = img.repeat(1 , 3, 1, 1)
        img = img.flip(1) # rgb to bgr
        img = (img + 1)/2 * 256  # since img is between -1 and +1
        img -= mean_bgr
        return img/128.0

    def __init__(self, loss=nn.L1Loss()):
        super().__init__()
        self.criterion = loss
        self.contentFunc = self.getContentFunc()

    def forward(self, x, y):
        x = self.preprocess(x)
        y = self.preprocess(y)

        f_x = self.contentFunc(x)
        f_y = self.contentFunc(y)
        f_y_no_grad = f_y.detach()
        loss = self.criterion(f_x, f_y_no_grad)
        return loss


class Weighted_MSE_loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__()
        L = np.load("../dataset/zernike_psfs/L.npy").astype(np.float32)  # todo: take this from args
        self.L = torch.from_numpy(L)

    def forward(self, x, y):
        return torch.mean(torch.pow(x - y, 2) @ self.L.to(x.device))


class PSNR_loss(_Loss):
    # def __init__(self, **kwargs):
    #     super(PSNR_loss, self).__init__()
    #     self.intensity_max = 1.0  # should be max^2. If image values are 0-255, 255^2

    # def forward(self, x, gt):
    #     # raise Exception("PSNR LOSS NO GRADIENT AFTER CALCULATION")
    #     # if x.max() > 1.0 or gt.max() > 1.0 or x.min() < 0 or gt.min() < 0:
    #     #     print(f"Clipping 0-1 when taking PSNR. {x.min():.2f}<x<{x.max():.2f} {gt.min():.2f}<gt<{gt.max():.2f}")
    #     x = torch.clip(x, 0.0, 1.0)
    #     gt = torch.clip(gt, 0.0, 1.0)
    #     mse = F.mse_loss(x, gt, reduction='none')
    #     mse_split = torch.split(mse, 1, dim=0)
    #     mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    #     psnr_list = [10.0 * math.log10(self.intensity_max / mse) for mse in mse_list]
    #     return torch.mean(torch.Tensor(psnr_list))
    def __init__(self, **kwargs):
        super(PSNR_loss, self).__init__()
        self.scale = 10 / np.log(10)
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        return self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class SSIM_loss(_Loss):
    def __init__(self, **kwargs):
        super(SSIM_loss, self).__init__()

    def forward(self, x, gt):
        return 1 - ssim(x, gt)


def loss_parse(loss_str):
    """
    parse loss parameters
    """
    ratios = []
    losses = []
    str_temp = loss_str.split('|')
    for item in str_temp:
        substr_temp = item.split('*')
        ratios.append(float(substr_temp[0]))
        losses.append(substr_temp[1])
    return ratios, losses


class Loss(nn.Module):
    """
    Training loss
    """

    def __init__(self, loss_str, **kwargs):
        super(Loss, self).__init__()
        # if args.padding > 0:
        #     print(f"Loss crops the padding of size {args.padding}")
        # self.padding = args.padding
        self.loss_str = loss_str
        ratios, losses = loss_parse(loss_str)
        self.losses_name = losses
        self.ratios = ratios
        self.losses = []
        for loss in losses:
            loss_fn = eval(f'{loss}(**kwargs)')
            self.losses.append(loss_fn)

    def forward(self, x, y):
        losses = {}
        loss_all = 0
        # x = x[..., self.padding:-self.padding, self.padding:-self.padding]
        # y = y[..., self.padding:-self.padding, self.padding:-self.padding]
        for i in range(len(self.losses)):
            loss_sub = self.ratios[i] * self.losses[i](x, y)
            losses[self.losses_name[i]] = loss_sub
            loss_all += loss_sub
        losses['all'] = loss_all
        return losses


def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1 = torch.clamp(img1, min=0, max=1)
    img2 = torch.clamp(img2, min=0, max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda()
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
