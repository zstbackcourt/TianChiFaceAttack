#python3
# -*- coding: utf-8 -*-
# File  : test_new_idea.py
# Author: Wang Chao
# Date  : 2019-09-01



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

import torch.nn.functional as F
from tqdm import *

warnings.filterwarnings('ignore')

import sys

sys.path.append('../')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from Face_recognition.inception_resnet_v1.inception_resnet_v1 import InceptionResnetV1
from Face_recognition.irse.model_irse import IR_50, IR_152
from Face_recognition.insightface.insightface import Backbone,MobileFaceNet

# ##### on swj's server
# from face_recognition.inception_resnet_v1.inception_resnet_v1 import InceptionResnetV1
# from face_recognition.irse.model_irse import IR_50, IR_152
# import insightface
# from face_recognition.insightface.insightface import Backbone, MobileFaceNet
# from face_recognition.vgg16.vgg16 import CenterLossModel, loadCheckpoint
# from face_recognition.resnet34_triplet.resnet34 import Resnet34Triplet




def img2tensor(img):
    img = img.swapaxes(1, 2).swapaxes(0, 1)
    img = np.reshape(img, [1, 3, 112, 112])
    img = np.array(img, dtype=np.float32)
    img = (img - 127.5) / 128.0
    img = torch.from_numpy(img)
    return img


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


class GaussianBlur(nn.Module):
    def __init__(self, kernel):
        super(GaussianBlur, self).__init__()
        self.kernel_size = len(kernel)
        # print('kernel size is {0}.'.format(self.kernel_size))
        assert self.kernel_size % 2 == 1, 'kernel size must be odd.'

        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze_(1)
        x2 = x[:, 1, :, :].unsqueeze_(1)
        x3 = x[:, 2, :, :].unsqueeze_(1)
        padding = self.kernel_size // 2
        x1 = F.conv2d(x1, self.weight, padding=padding)
        x2 = F.conv2d(x2, self.weight, padding=padding)
        x3 = F.conv2d(x3, self.weight, padding=padding)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def main():
    sample_dir = './target_mean_face_1/'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    InceptionResnet_model_1 = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print('load InceptionResnet-vggface2.pt successfully')

    InceptionResnet_model_2 = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    print('load InceptionResnet-casia-webface.pt successfully')

    IR_50_model_1 = IR_50([112, 112])
    IR_50_model_1.load_state_dict(
        torch.load(
            '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/Face_recognition/irse/model/backbone_ir50_asia.pth'))
    IR_50_model_1.eval().to(device)
    print('load IR_50 successfully')

    IR_152_model_1 = IR_152([112, 112])
    IR_152_model_1.load_state_dict(
        torch.load(
            '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/Face_recognition/irse/model/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'))
    IR_152_model_1.eval().to(device)
    print('load IR_152 successfully')

    IR_SE_50 = Backbone(50,mode='ir_se').eval().to(device)
    print('load IR_SE_50 successfully')

    mobileFaceNet = MobileFaceNet(512).eval().to(device)
    print('load mobileFaceNet successfully')

    # IR_152_model_2 = IR_152([112, 112])
    # IR_152_model_2.load_state_dict(
    #     torch.load(
    #         '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/Face_recognition/irse/model/Head_ArcFace_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'))
    # IR_152_model_2.eval().to(device)
    # print('load IR_152_ArcFace successfully')

    import insightface


    Insightface_iresnet34 = insightface.iresnet34(pretrained=True)
    Insightface_iresnet34.eval().to(device)
    print('load Insightface_iresnet34 successfully')

    Insightface_iresnet50 = insightface.iresnet50(pretrained=True)
    Insightface_iresnet50.eval().to(device)
    print('load Insightface_iresnet50 successfully')

    Insightface_iresnet100 = insightface.iresnet100(pretrained=True)
    Insightface_iresnet100.eval().to(device)
    print('load Insightface_iresnet100 successfully')

###########################vgg16
    from Face_recognition.vgg16.vgg16 import CenterLossModel,loadCheckpoint
    vgg16_checkpoint=loadCheckpoint('/notebooks/Workspace/tmp/pycharm_project_314/TianChi/Face_recognition/vgg16/model');

    VGG16 = CenterLossModel(embedding_size=512,num_classes=712,checkpoint=vgg16_checkpoint).eval().to(device)
    print('load VGG16 successfully')



    # ################on swj's server
    # InceptionResnet_model_1 = InceptionResnetV1(pretrained='vggface2').eval()
    # print('load InceptionResnet-vggface2.pt successfully')
    #
    # InceptionResnet_model_2 = InceptionResnetV1(pretrained='casia-webface').eval()
    # print('load InceptionResnet-casia-webface.pt successfully')
    #
    # IR_50_model_1 = IR_50([112, 112])
    # IR_50_model_1.load_state_dict(torch.load('./face_recognition/irse/model/backbone_ir50_asia.pth'))
    # IR_50_model_1.eval()
    # print('load IR_50 successfully')
    #
    # IR_152_model_1 = IR_152([112, 112])
    # IR_152_model_1.load_state_dict(torch.load(
    #     './face_recognition/irse/model/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'))
    # IR_152_model_1.eval()
    # print('load IR_152 successfully')
    #
    # IR_SE_50 = Backbone(50, mode='ir_se').eval()
    # print('load IR_SE_50 successfully')
    #
    # mobileFaceNet = MobileFaceNet(512).eval()
    # print('load mobileFaceNet successfully')
    #
    # Insightface_iresnet34 = insightface.iresnet34(pretrained=True)
    # Insightface_iresnet34.eval()
    # print('load Insightface_iresnet34 successfully')
    #
    # Insightface_iresnet50 = insightface.iresnet50(pretrained=True)
    # Insightface_iresnet50.eval()
    # print('load Insightface_iresnet50 successfully')
    #
    # Insightface_iresnet100 = insightface.iresnet100(pretrained=True)
    # Insightface_iresnet100.eval()
    # print('load Insightface_iresnet100 successfully')
    #
    # vgg16_checkpoint = loadCheckpoint('./face_recognition/vgg16/model')
    # VGG16 = CenterLossModel(embedding_size=512, num_classes=712, checkpoint=vgg16_checkpoint).eval()
    # print('load vgg16 successfully')

    ####load model to cuda
    InceptionResnet_model_1.to(device)
    InceptionResnet_model_2.to(device)
    IR_50_model_1.to(device)
    IR_152_model_1.to(device)
    IR_SE_50.to(device)
    mobileFaceNet.to(device)
    Insightface_iresnet34.to(device)
    Insightface_iresnet50.to(device)
    Insightface_iresnet100.to(device)
    VGG16.to(device)

    criterion = nn.MSELoss()
    # cpu
    # collect all images to attack
    paths = []
    picpath = '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/images'
    for root, dirs, files in os.walk(picpath):
        for f in files:
            paths.append(os.path.join(root, f))
    random.shuffle(paths)

    # paras
    eps = 1
    steps = 50
    output_path = './output_img'
    momentum = 1.0
    alpha = 0.35
    beta = 0.9
    gamma = 0.1

    #######cal mean feature face
    print('cal mean feature face #########################')
    mean_face = torch.zeros(512).detach().to(device)
    for path in tqdm(paths):
        start = time.time()
        print('cal mean face ' + path + '  ===============>')
        image = Image.open(path)

        # define paras
        # in_tensor is origin tensor of image
        # in_variable changes with gradient

        in_tensor_1 = img2tensor(np.array(image))
        # print(in_tensor.shape)
        in_variable_1 = in_tensor_1.detach().to(device)
        in_tensor_1 = in_tensor_1.squeeze().to(device)
        this_feature_face = None

        # # origin feature

        _origin_InceptionResnet_model_1 = InceptionResnet_model_1(in_variable_1).volatile = True
        _origin_InceptionResnet_model_2 = InceptionResnet_model_2(in_variable_1).volatile = True
        _origin_IR_50_model_1 = IR_50_model_1(in_variable_1).volatile = True
        _origin_IR_152_model_1 = IR_152_model_1(in_variable_1).volatile = True
        _origin_IR_SE_50 = IR_SE_50(in_variable_1).volatile = True
        _origin_mobileFaceNet = mobileFaceNet(in_variable_1).volatile = True
        _origin_Insightface_iresent34 = Insightface_iresnet34(in_variable_1).volatile = True
        _origin_Insightface_iresent50 = Insightface_iresnet50(in_variable_1).volatile = True
        _origin_Insightface_iresent100 = Insightface_iresnet100(in_variable_1).volatile = True
        _origin_VGG16 = VGG16.forward_GetFeature(in_variable_1).volatile = True

        this_feature_face = _origin_InceptionResnet_model_1*0.7 + \
                            _origin_InceptionResnet_model_2*0.7 + \
                            _origin_IR_50_model_1 *0.8+ \
                            _origin_IR_152_model_1 *0.8 + \
                            _origin_IR_SE_50 *0.7+ \
                            _origin_mobileFaceNet *0.7+ \
                            _origin_Insightface_iresent34 *0.8 + \
                            _origin_Insightface_iresent50 *0.9 + \
                            _origin_Insightface_iresent100 *0.9 + \
                            _origin_VGG16 *0.7

        this_feature_face = this_feature_face / 10.
        mean_face = mean_face + this_feature_face

        del _origin_InceptionResnet_model_1
        del _origin_InceptionResnet_model_2
        del _origin_IR_50_model_1
        del _origin_IR_152_model_1
        del _origin_IR_SE_50
        del _origin_mobileFaceNet
        del _origin_Insightface_iresent34
        del _origin_Insightface_iresent50
        del _origin_Insightface_iresent100
        del _origin_VGG16
        del this_feature_face
        del in_tensor_1
        del in_variable_1

    mean_face = mean_face / 712.
    print('finish cal mean face...')
    #############################

    #####
    print('######attack...##################')
    for path in tqdm(paths):

        start = time.time()
        print('processing ' + path + '  ===============>')
        image = Image.open(path)

        # define paras
        # in_tensor is origin tensor of image
        # in_variable changes with gradient

        in_tensor = img2tensor(np.array(image))
        origin_variable = in_tensor.detach()
        origin_variable = origin_variable.to(device)
        tar_tensor = mean_face.to(device)
        in_variable = in_tensor.detach()
        in_variable = in_variable.to(device)
        tar_variable = tar_tensor.detach()
        tar_variable = tar_variable.to(device)
        in_tensor = in_tensor.squeeze()
        in_tensor = in_tensor.to(device)
        adv = None

        perturbation = torch.Tensor(3, 112, 112).uniform_(-0.05, 0.05)
        perturbation = perturbation.to(device)
        in_variable += perturbation
        in_variable.data.clamp_(-1.0, 1.0)
        in_variable.requires_grad = True
        g_noise = torch.zeros_like(in_variable)
        g_noise = g_noise.to(device)

        origin_InceptionResnet_model_1 = InceptionResnet_model_1(origin_variable)
        origin_InceptionResnet_model_2 = InceptionResnet_model_2(origin_variable)
        origin_IR_50_model_1 = IR_50_model_1(origin_variable)
        origin_IR_152_model_1 = IR_152_model_1(origin_variable)
        origin_IR_SE_50 = IR_SE_50(origin_variable)
        origin_mobileFaceNet = mobileFaceNet(origin_variable)
        # # origin_IR_152_model_2 = IR_152_model_2(in_variable)
        origin_Insightface_iresent34 = Insightface_iresnet34(origin_variable)
        origin_Insightface_iresent50 = Insightface_iresnet50(origin_variable)
        origin_Insightface_iresent100 = Insightface_iresnet100(origin_variable)
        origin_VGG16 = VGG16.forward_GetFeature(origin_variable)

        #  sum gradient
        for i in range(steps):
            print('step: ' + str(i))
            mediate_InceptionResnet_model_1 = InceptionResnet_model_1(in_variable)
            mediate_InceptionResnet_model_2 = InceptionResnet_model_2(in_variable)
            mediate_IR_50_model_1 = IR_50_model_1(in_variable)
            mediate_IR_152_model_1 = IR_152_model_1(in_variable)
            mediate_IR_SE_50 = IR_SE_50(in_variable)
            mediate_mobileFaceNet = mobileFaceNet(in_variable)
            # # origin_IR_152_model_2 = IR_152_model_2(in_variable)
            mediate_Insightface_iresent34 = Insightface_iresnet34(in_variable)
            mediate_Insightface_iresent50 = Insightface_iresnet50(in_variable)
            mediate_Insightface_iresent100 = Insightface_iresnet100(in_variable)
            mediate_VGG16 = VGG16.forward_GetFeature(in_variable)

            average_out = (mediate_InceptionResnet_model_1+mediate_InceptionResnet_model_2+mediate_IR_50_model_1+\
               mediate_IR_152_model_1+mediate_IR_SE_50+mediate_mobileFaceNet+mediate_Insightface_iresent34+\
               mediate_Insightface_iresent50+mediate_Insightface_iresent100+mediate_VGG16)/10

            # loss1 far away from orgin image, loss2 approach target image
            # loss1 = criterion(origin_InceptionResnet_model_1, mediate_InceptionResnet_model_1) + \
            #         criterion(origin_InceptionResnet_model_2, mediate_InceptionResnet_model_2) + \
            #         criterion(origin_IR_50_model_1, mediate_IR_50_model_1) + \
            #         criterion(origin_IR_SE_50, mediate_IR_SE_50) + \
            #         criterion(origin_mobileFaceNet, mediate_mobileFaceNet) + \
            #         criterion(origin_Insightface_iresent34, mediate_Insightface_iresent34) + \
            #         criterion(origin_Insightface_iresent50, mediate_Insightface_iresent50) + \
            #         criterion(origin_Insightface_iresent100, mediate_Insightface_iresent100) + \
            #         criterion(origin_VGG16, mediate_VGG16)

            loss1 = criterion(origin_InceptionResnet_model_1, mediate_InceptionResnet_model_1) * 0.7 + \
                    criterion(origin_InceptionResnet_model_2, mediate_InceptionResnet_model_2) * 0.7 + \
                    criterion(origin_IR_50_model_1, mediate_IR_50_model_1) * 0.8 + \
                    criterion(origin_IR_152_model_1, mediate_IR_152_model_1) * 0.8 + \
                    criterion(origin_IR_SE_50, mediate_IR_SE_50) * 0.7 + \
                    criterion(origin_mobileFaceNet, mediate_mobileFaceNet) * 0.7 + \
                    criterion(origin_Insightface_iresent34, mediate_Insightface_iresent34) * 0.8 + \
                    criterion(origin_Insightface_iresent50, mediate_Insightface_iresent50) * 0.9 + \
                    criterion(origin_Insightface_iresent100, mediate_Insightface_iresent100) * 0.9 + \
                    criterion(origin_VGG16, mediate_VGG16) * 0.7


            loss2 = criterion(mediate_InceptionResnet_model_1, mean_face) *0.7+ \
                    criterion(mediate_InceptionResnet_model_2, mean_face) *0.7+ \
                    criterion(mediate_IR_50_model_1, mean_face) *0.8+ \
                    criterion(mediate_IR_152_model_1, mean_face) *0.8+\
                    criterion(mediate_IR_SE_50, mean_face) *0.7+ \
                    criterion(mediate_mobileFaceNet, mean_face) *0.7+ \
                    criterion(mediate_Insightface_iresent34, mean_face) *0.8+ \
                    criterion(mediate_Insightface_iresent50, mean_face) *0.9+ \
                    criterion(mediate_Insightface_iresent100, mean_face)*0.9 + \
                    criterion(mediate_VGG16, mean_face)*0.7


            loss3 = criterion(average_out,mediate_InceptionResnet_model_1)+ \
                    criterion(average_out,mediate_InceptionResnet_model_2)+ \
                    criterion(average_out,mediate_IR_50_model_1) + \
                    criterion(average_out,mediate_IR_152_model_1) + \
                    criterion(average_out,mediate_mobileFaceNet) + \
                    criterion(average_out,mediate_Insightface_iresent34)+ \
                    criterion(average_out,mediate_Insightface_iresent50) + \
                    criterion(average_out,mediate_Insightface_iresent100)+ \
                    criterion(average_out,mediate_VGG16)+ \
                    criterion(average_out,mediate_IR_SE_50)

            loss = alpha * loss1 - beta* loss2 - gamma*loss3

            print('loss : %f' % loss)
            print('loss1 : %f' % loss1)
            print('loss2 : %f' % loss2)
            # compute gradients
            loss.backward(retain_graph=True)

            g_noise = momentum * g_noise + (in_variable.grad / in_variable.grad.data.norm(1))*0.9
            g_noise = g_noise / g_noise.data.norm(1)

            g1 = g_noise
            g2 = g_noise

            # if i % 3 == 0 :
            kernel = gkern(3, 2).astype(np.float32)
            gaussian_blur1 = GaussianBlur(kernel)
            gaussian_blur1.to(device)
            g1 = gaussian_blur1(g1)
            g1 = torch.clamp(g1, -0.2, 0.2)
            # else:
            addition = TVLoss()
            addition.to(device)
            g2 = addition(g2)

            g_noise = 0.25 * g1 + 0.75 * g2
            g_noise.clamp_(-0.05, 0.05)

            in_variable.data = in_variable.data + (
                        (eps / 255.) * torch.sign(g_noise))  # * torch.from_numpy(mat).unsqueeze(0).float()

            in_variable.grad.data.zero_()  # unnecessary

            # g_noise = in_variable.data - origin_variable
            # g_noise.clamp_(-0.2, 0.2)
            # in_variable.data = origin_variable + g_noise

        # deprocess image
        adv = in_variable.data.cpu().numpy()[0]  # (3, 112, 112)
        perturbation = (adv - in_tensor.cpu().numpy())

        adv = adv * 128.0 + 127.0
        adv = adv.swapaxes(0, 1).swapaxes(1, 2)
        adv = adv[..., ::-1]
        adv = np.clip(adv, 0, 255).astype(np.uint8)

        # sample_dir = './target_mean_face/'
        # if not os.path.exists(sample_dir):
        #     os.makedirs(sample_dir)

        advimg = sample_dir + path.split('/')[-1].split('.')[-2] + '.jpg'

        cv2.imwrite(advimg, adv)
        print("save path is " + advimg)
        print('cost time is %.2f s ' % (time.time() - start))


if __name__ == '__main__':
    main()

