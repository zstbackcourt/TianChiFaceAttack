import os
import sys
import time
import torch
import cv2
import warnings
import random
import numpy as np
import argparse
from PIL import Image
import scipy.stats as st
from torchvision import models
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_facenet import model_920, model_921


IMG_SIZE = 224

#  ensemble multi-model
# 1
model_ir50_epoch120 = IR_50([112,112])
model_ir50_epoch120.load_state_dict(torch.load('./Defense_Model/backbone_ir50_ms1m_epoch120.pth',map_location='cuda'))
model_ir50_epoch120.eval()
criterion_ir50_epoch120 = nn.MSELoss()
# 2
model_IR_152_Epoch_112 = IR_152([112,112])
model_IR_152_Epoch_112.load_state_dict(torch.load('./Defense_Model/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth',map_location='cuda'))
model_IR_152_Epoch_112.eval()
criterion_IR_152_Epoch_112 = nn.MSELoss()
# 3
model_IR_SE_50_Epoch_2 = IR_SE_50([112,112])
model_IR_SE_50_Epoch_2.load_state_dict(torch.load('./Defense_Model/Backbone_IR_SE_50_Epoch_2_Batch_45488_Time_2019-08-03-19-39_checkpoint.pth',map_location='cuda'))
model_IR_SE_50_Epoch_2.eval()
criterion_IR_SE_50_Epoch_2 = nn.MSELoss()
# 4
model_IR_SE_152_Epoch_4 = IR_SE_152([112,112])
model_IR_SE_152_Epoch_4.load_state_dict(torch.load('./Defense_Model/Backbone_IR_SE_152_Epoch_4_Batch_181956_Time_2019-08-06-07-29_checkpoint.pth',map_location='cuda'))
model_IR_SE_152_Epoch_4.eval()
criterion_IR_SE_152_Epoch_4 = nn.MSELoss()
# 5
model_ResNet_101_Epoch_4 = ResNet_101([112, 112])
model_ResNet_101_Epoch_4.load_state_dict(
    torch.load('./Defense_Model/Backbone_ResNet_101_Epoch_4_Batch_90976_Time_2019-08-04-11-34_checkpoint.pth',
               map_location='cuda'))
model_ResNet_101_Epoch_4.eval()
criterion_ResNet_101_Epoch_4 = nn.MSELoss()
# 6
model_ResNet_152_Epoch_1 = ResNet_152([112, 112])
model_ResNet_152_Epoch_1.load_state_dict(
    torch.load('./Defense_Model/Backbone_ResNet_152_Epoch_1_Batch_22744_Time_2019-08-03-01-01_checkpoint.pth',
               map_location='cuda'))
model_ResNet_152_Epoch_1.eval()
criterion_ResNet_152_Epoch_1 = nn.MSELoss()
# 7
model_ResNet_50_Epoch_3 = ResNet_50([112, 112])
model_ResNet_50_Epoch_3.load_state_dict(
    torch.load('./Defense_Model/Backbone_ResNet_50_Epoch_3_Batch_34116_Time_2019-08-02-19-12_checkpoint.pth',
               map_location='cuda'))
model_ResNet_50_Epoch_3.eval()
criterion_ResNet_50_Epoch_3 = nn.MSELoss()

criterion = nn.MSELoss()

# cpu
# collect all images to attack
paths = []
picpath = os.getcwd() + '/raw_faces'
dire = None
for root, dirs, files in os.walk(picpath):
    if dirs and dirs != ['.ipynb_checkpoints']:
        dire = dirs
    for f in files:
        paths.append(os.path.join(root, f))
random.shuffle(paths)

# paras
eps = 1
steps = 20
output_path = './output_img'
momentum = 1.0

for path in paths:
    start = time.time()
    print('processing ' + path + '  ===============>')
    image = Image.open(path)

    # define paras
    # in_tensor is origin tensor of image
    # in_variable changes with gradient
    in_tensor = img2tensor(np.array(image))
    print(in_tensor.shape)
    in_variable = in_tensor.detach()
    in_tensor = in_tensor.squeeze()
    adv = None
    # origin feature
    origin_feat_ir50_epoch120 = model_ir50_epoch120(in_variable)
    origin_IR_152_Epoch_112 = model_IR_152_Epoch_112(in_variable)
    origin_IR_SE_50_Epoch_2 = model_IR_SE_50_Epoch_2(in_variable)
    origin_IR_SE_152_Epoch_4 = model_IR_SE_152_Epoch_4(in_variable)
    origin_ResNet_101_Epoch_4 = model_ResNet_101_Epoch_4(in_variable)
    origin_ResNet_152_Epoch_1 = model_ResNet_152_Epoch_1(in_variable)
    origin_ResNet_50_Epoch_3 = model_ResNet_50_Epoch_3(in_variable)

    # 1. untarget attack -> random noise
    # 2. target attack -> x = alpha * target + (1 - alpha) * x
    perturbation = torch.Tensor(3, 112, 112).uniform_(-0.1, 0.1)
    in_variable += perturbation
    in_variable.data.clamp_(-1.0, 1.0)
    in_variable.requires_grad = True
    g_noise = 0.0

    #  sum gradient
    for i in range(steps):
        print('step: ' + str(i))
        # in_variable = in_variable.to(device)
        out_feat_ir50_epoch120 = model_ir50_epoch120(in_variable)
        out_IR_152_Epoch_112 = model_IR_152_Epoch_112(in_variable)
        out_IR_SE_50_Epoch_2 = model_IR_SE_50_Epoch_2(in_variable)
        out_IR_SE_152_Epoch_4 = model_IR_SE_152_Epoch_4(in_variable)
        out_ResNet_101_Epoch_4 = model_ResNet_101_Epoch_4(in_variable)
        out_ResNet_152_Epoch_1 = model_ResNet_152_Epoch_1(in_variable)
        out_ResNet_50_Epoch_3 = model_ResNet_50_Epoch_3(in_variable)

        # loss = criterion(origin_feat_ir50_epoch120, out_feat_ir50_epoch120)
        loss = criterion(origin_feat_ir50_epoch120, out_feat_ir50_epoch120) + criterion(origin_IR_152_Epoch_112,
                                                                                        out_IR_152_Epoch_112) + criterion(
            origin_IR_SE_50_Epoch_2, out_IR_SE_50_Epoch_2) + criterion(origin_IR_SE_152_Epoch_4,
                                                                       out_IR_SE_152_Epoch_4) + criterion(
            origin_ResNet_101_Epoch_4, out_ResNet_101_Epoch_4) + criterion(origin_ResNet_152_Epoch_1,
                                                                           out_ResNet_152_Epoch_1) + criterion(
            origin_ResNet_50_Epoch_3, out_ResNet_50_Epoch_3)

        print('loss : %f' % loss)
        # compute gradients
        loss.backward(retain_graph=True)

        g_noise = momentum * g_noise + (in_variable.grad / in_variable.grad.data.norm(1))
        g_noise = g_noise / g_noise.data.norm(1)

        if i % 2 == 0:
            kernel = gkern(3, 2).astype(np.float32)
            gaussian_blur1 = GaussianBlur(kernel)
            g_noise = gaussian_blur1(g_noise)
            g_noise = torch.clamp(g_noise, -0.1, 0.1)
        else:
            addition = TVLoss()
            g_noise = addition(g_noise)

        in_variable.data = in_variable.data + (
                    (eps / 255.) * torch.sign(g_noise))  # * torch.from_numpy(mat).unsqueeze(0).float()

        in_variable.grad.data.zero_()  # unnecessary

        # deprocess image
    adv = in_variable.data.cpu().numpy()[0]  # (3, 112, 112)
    perturbation = (adv - in_tensor.numpy())

    adv = adv * 128.0 + 127.0
    adv = adv.swapaxes(0, 1).swapaxes(1, 2)
    adv = adv[..., ::-1]
    adv = np.clip(adv, 0, 255).astype(np.uint8)

    id = 100
    advimg = './output_img/' + path.split('/')[-1].split('.')[-2] + '_' + str(id) + '.jpg'
    id += 1

    cv2.imwrite(advimg, adv)
    print("save path is " + advimg)
    print('cost time is %.2f秒 ' % (time.time() - start))