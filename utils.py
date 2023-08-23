import numpy as np
import random
import torch
import torchvision
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
from model import *
from Resnet import *
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset import config, Dataset, collate_fn

def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1

    return float(lr / 2 * cos_out)

def load_model(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet50_pmg':
        net = resnet50(pretrained=pretrain)
        for param in net.parameters():
            param.requires_grad = require_grad
        net = PC(net, 512, 200)

    return net


def DSLM(feature):
    b, c, w, h = feature.shape

    mask = torch.zeros([b, w, h]).cuda()
    for index in range(b):
        feature_maps = feature[index]

        weights = torch.mean(torch.mean(feature_maps, dim=-2, keepdim=True), dim=-1, keepdim=True)

        cam = torch.sum(weights * feature_maps, dim=0)
        cam = torch.clamp(cam, 0)
        cam = cam - torch.min(cam)
        cam = cam / torch.max(cam)
        mask[index, :, :] = cam

    wt = torch.zeros_like(cam).cuda()

    for index in range(b):
        mean_b = torch.mean(cam[index])
        std_b = torch.std(cam[index])
        wt[index, :, :] = (cam[index] - mean_b) / std_b

    ac_map = torch.clamp(wt, 0)
    return ac_map

def SGLoss(fs, ft):

    mask_s = DSLM(fs)
    mask_t = DSLM(ft)

    loss = torch.mean(mask_s * mask_t)
    return loss
