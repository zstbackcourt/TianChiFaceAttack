#python3
# -*- coding: utf-8 -*-


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
import csv

warnings.filterwarnings('ignore')

import sys

sys.path.append('../')


from Face_recognition.inception_resnet_v1.inception_resnet_v1 import InceptionResnetV1
from Face_recognition.irse.model_irse import IR_50, IR_152
from Face_recognition.insightface.insightface import Backbone,MobileFaceNet
import insightface
from Face_recognition.arcface.arcface import Arcface


device_0 = torch.device('cuda:0')
device_1 = torch.device('cuda:1')


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



def input_diversity(image_tensor,image_size,prob):
    np.random.seed(1234)
    # rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
    rnd = np.random.randint(80,111)
    rescaled = nn.Upsample(scale_factor=rnd/image_size, mode='nearest')(image_tensor)
    h_rem = image_size - rnd
    w_rem = image_size - rnd

    pad_top = np.random.randint(0,h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = np.random.randint(0,w_rem)
    pad_right = w_rem - pad_left
    dim = (pad_left, pad_right, pad_top, pad_bottom)

    padded = F.pad(rescaled,dim,"constant",value=0.)

    padded.view(image_tensor.shape[0], image_size, image_size, 3)
    if np.random.random() <prob:
        return padded
    else:
        return image_tensor



'''
本想在在input_diversity 上多加点变换
但是pytorch的transforms.ToPILImage会截断梯度（里面有numpy）
所以只能用手写的tensor的resize ...
'''''
def myRandomTransform(image_tensor):
    Crops = [
        transforms.CenterCrop(85),
        transforms.RandomCrop(85),
        transforms.RandomResizedCrop(85, scale=(0.7, 1.0))
    ]
    myTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomChoice(Crops),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomApply([transforms.RandomRotation(30, resample=False, expand=True, center=None)], p=0.5),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=0.5),
        transforms.Resize(112),
        transforms.ToTensor()

    ])
    new_image_tensor = image_tensor
    for i in range(len(image_tensor)):
        new_image_tensor[i] = myTransform(image_tensor[i])
    return new_image_tensor


'''
因为TianChi这次是在LFW上抽了712张照片做验证
所以在提不动点之后...想尝试引入更多的额外数据计算平均脸
'''
def cal_mean_face_by_extend_dataset(models):
    mean_face = torch.zeros(512)
    expand_dataset = os.listdir("/notebooks/Workspace/tmp/pycharm_project_314/TianChi/lfw-aligned")
    with open("/notebooks/Workspace/tmp/pycharm_project_314/TianChi/securityAI_round1_dev.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in tqdm(reader):
            if line[1] == 'ImageName':
                continue
            if line[2] not in expand_dataset:
                print('can not find ', line[2])
            else:
                eachdirpath = "/notebooks/Workspace/tmp/pycharm_project_314/TianChi/lfw-aligned/" + line[2]
                pics = os.listdir(eachdirpath)
                pics = [eachdirpath + '/' + i for i in pics]
                this_person_face_feature = cal_expand_tensor(pics, models)
                mean_face = mean_face + this_person_face_feature
        mean_face = mean_face /712.

    return mean_face

def cal_expand_tensor(pics, models):
    pic_mean_face = torch.zeros(512)
    with torch.no_grad():
        for pic in pics:
            image = Image.open(pic)
            image = image.resize([112, 112], Image.NEAREST).convert("RGB")
            in_tensor = img2tensor(np.array(image))
            _origin_InceptionResnet_model_1 = models[0](in_tensor.to(device_0)).cpu()

            _origin_IR_50_model_1 = models[1](in_tensor.to(device_0)).cpu()

            _origin_Insightface_iresent34 = models[2](in_tensor.to(device_1)).cpu()

            _origin_Insightface_iresent50 = models[3](in_tensor.to(device_1)).cpu()

            _origin_Insightface_iresent100 = models[4](in_tensor.to(device_1)).cpu()

            _origin_arcface = models[5](in_tensor.to(device_0)).cpu()

            this_feature_face = (_origin_InceptionResnet_model_1 + \
                                 _origin_IR_50_model_1 + \
                                 _origin_Insightface_iresent34 + \
                                 _origin_Insightface_iresent50 + \
                                 _origin_Insightface_iresent100 + \
                                 _origin_arcface) / 6
            pic_mean_face = pic_mean_face + this_feature_face

    pic_mean_face = pic_mean_face / len(pics)
    return pic_mean_face


'''
最优成绩24 /13.90 只用了六个模型
貌似我找到的模型本身对于这个任务的能力都不算强
最强的Insightface_iresnet100 + all trick 之后就可以到14.6

'''

'''
Used to crop the image to prevent pixel disturbances from exceeding the limit
'''

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()

    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result


'''
main function 
'''
def main():
    sample_dir = './test_DI-2-FGSM-3/'
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)


    InceptionResnet_model_1 = InceptionResnetV1(pretrained='vggface2').eval().to(device_0)
    print('load InceptionResnet-vggface2.pt successfully')

    # InceptionResnet_model_2 = InceptionResnetV1(pretrained='casia-webface').eval().to(device_0)
    # print('load InceptionResnet-casia-webface.pt successfully')

    IR_50_model_1 = IR_50([112, 112])
    IR_50_model_1.load_state_dict(
        torch.load(
            '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/Face_recognition/irse/model/backbone_ir50_asia.pth'))
    IR_50_model_1.eval().to(device_0)
    print('load IR_50 successfully')

#     IR_152_model_1 = IR_152([112, 112])
#     IR_152_model_1.load_state_dict(
#         torch.load(
#             '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/Face_recognition/irse/model/Backbone_IR_152_Epoch_112_Batch_2547328_Time_2019-07-13-02-59_checkpoint.pth'))
#     IR_152_model_1.eval().to(device_0)
#     print('load IR_152 successfully')

#     IR_SE_50 = Backbone(50,mode='ir_se').eval().to(device_1)
#     print('load IR_SE_50 successfully')

#     mobileFaceNet = MobileFaceNet(512).eval().to(device_0)
#     print('load mobileFaceNet successfully')



    Insightface_iresnet34 = insightface.iresnet34(pretrained=True)
    Insightface_iresnet34.eval().to(device_1)
    print('load Insightface_iresnet34 successfully')

    Insightface_iresnet50 = insightface.iresnet50(pretrained=True)
    Insightface_iresnet50.eval().to(device_1)
    print('load Insightface_iresnet50 successfully')

    Insightface_iresnet100 = insightface.iresnet100(pretrained=True)
    Insightface_iresnet100.eval().to(device_1)
    print('load Insightface_iresnet100 successfully')

# ##########################vgg16
#     from Face_recognition.vgg16.vgg16 import CenterLossModel,loadCheckpoint
#     vgg16_checkpoint=loadCheckpoint('/notebooks/Workspace/tmp/pycharm_project_314/TianChi/Face_recognition/vgg16/model');
#
#     VGG16 = CenterLossModel(embedding_size=512,num_classes=712,checkpoint=vgg16_checkpoint).eval().to(device_1)
#     print('load VGG16 successfully')

    arc_face_ir_se_50 = Arcface()
    arc_face_ir_se_50.eval()
    arc_face_ir_se_50.to(device_0)

    models = []
    models.append(InceptionResnet_model_1)
    models.append(IR_50_model_1)
    models.append(Insightface_iresnet34)
    models.append(Insightface_iresnet50)
    models.append(Insightface_iresnet100)
    models.append(arc_face_ir_se_50)


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
    momentum = 0.3
    alpha = 0.35
    beta = 0.3
    gamma = 0.1



    #####cal mean feature face
    print('cal mean feature face #########################')

# ##########
# cal mean feature face on only 712 images
#     mean_face_1 = torch.zeros(1,3,112,112).detach()
#     for path in tqdm(paths):
#         image = Image.open(path)
#         in_tensor_1 = img2tensor(np.array(image))
#         mean_face_1 += in_tensor_1
#     mean_face_1 = mean_face_1 / 712
# ##########    
    
#     with torch.no_grad():
#         mean_face = torch.zeros(512).detach()
#         for path in tqdm(paths):
#             start = time.time()
#             print('cal mean face ' + path + '  ===============>')
#             image = Image.open(path)
#
#
#             # define paras
#             # in_tensor is origin tensor of image
#             # in_variable changes with gradient
#
#             in_tensor_1 = img2tensor(np.array(image))
#             # print(in_tensor.shape)
#             this_feature_face = None
#
#
#             # # origin feature
#
#             _origin_InceptionResnet_model_1 = InceptionResnet_model_1(in_tensor_1.to(device_0)).cpu()
#             ####################
# #             _origin_InceptionResnet_model_2 = InceptionResnet_model_2(in_tensor_1.to(device_0)).cpu()
#             ######################
# #             _origin_IR_50_model_1 = IR_50_model_1(in_tensor_1.to(device_0)).cpu()
#             ###########################
# #             _origin_IR_152_model_1 = IR_152_model_1(in_tensor_1.to(device_0)).cpu()
# #             _origin_IR_SE_50 = IR_SE_50(in_tensor_1.to(device_1)).cpu()
# #             _origin_mobileFaceNet = mobileFaceNet(in_tensor_1.to(device_0)).cpu()
#             #############################
#
#             _origin_Insightface_iresent34 = Insightface_iresnet34(in_tensor_1.to(device_1)).cpu()
#
#
#             _origin_Insightface_iresent50 = Insightface_iresnet50(in_tensor_1.to(device_1)).cpu()
#
#
#             _origin_Insightface_iresent100 = Insightface_iresnet100(in_tensor_1.to(device_1)).cpu()
#
#
#             _origin_arcface = arc_face_ir_se_50(in_tensor_1.to(device_0)).cpu()
#
#             ########################
# #             _origin_VGG16 = VGG16.forward_GetFeature(in_tensor_1.to(device_1)).cpu()
#             ########################
#
#             this_feature_face = _origin_InceptionResnet_model_1 + \
#                                 _origin_Insightface_iresent34  + \
#                                 _origin_Insightface_iresent50  + \
#                                 _origin_Insightface_iresent100  + \
#                                 _origin_arcface
#
# #             this_feature_face = _origin_InceptionResnet_model_1 + \
# #                                 _origin_InceptionResnet_model_2 +\
# #                                 _origin_IR_50_model_1 + \
# #                                 _origin_IR_152_model_1 +\
# #                                 _origin_IR_SE_50 +\
# #                                 _origin_mobileFaceNet +\
# #                                 _origin_Insightface_iresent34  + \
# #                                 _origin_Insightface_iresent50  + \
# #                                 _origin_Insightface_iresent100  + \
# #                                 _origin_arcface +\
# #                                 _origin_VGG16
#
#
#             this_feature_face = this_feature_face / 5.
#             mean_face = mean_face + this_feature_face
#
# #         del _origin_InceptionResnet_model_1
# #         del _origin_InceptionResnet_model_2
# #         del _origin_IR_50_model_1
# #         del _origin_IR_152_model_1
# #         del _origin_IR_SE_50
# #         del _origin_mobileFaceNet
# #         del _origin_Insightface_iresent34
# #         del _origin_Insightface_iresent50
# #         del _origin_Insightface_iresent100
# #         del _origin_VGG16
# #         del _origin_arcface
# #         del this_feature_face
# #         del in_tensor_1
#
#         del _origin_InceptionResnet_model_1
# #         del _origin_IR_50_model_1
#         del _origin_Insightface_iresent34
#         del _origin_Insightface_iresent50
#         del _origin_Insightface_iresent100
#         del _origin_arcface
#         del this_feature_face
#         del in_tensor_1
#
#     mean_face = mean_face / 712.

    mean_face = cal_mean_face_by_extend_dataset(models)
    print('finish cal mean face...')
    ############################

    print('######attack...##################')
    from mydataset import CustomDataset
    custom_dataset = CustomDataset()
    train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                               batch_size=6,
                                               shuffle=True)
    count = 0
    progressRate = 0.0
    for i, (x,path) in enumerate(train_loader):
        start = time.time()
        print('processing ' + str(progressRate) + '  ===============>')
        in_tensor = x
        origin_variable = in_tensor.detach()
        origin_variable = origin_variable

        in_variable = in_tensor.detach()
        in_variable = in_variable

        ##########
        in_variable_max = in_tensor.detach()
        in_variable_min = in_tensor.detach()
        in_variable_max = in_variable +0.1
        in_variable_min = in_variable -0.1
        ##########
        
        in_tensor = in_tensor.squeeze()
        in_tensor = in_tensor
        adv = None




        perturbation = torch.Tensor(x.size(0),3, 112, 112).uniform_(-0.05, 0.05) ###这里测试
        perturbation = perturbation
        in_variable += perturbation
        in_variable.data.clamp_(-1.0, 1.0)
        in_variable.requires_grad = True
        g_noise = torch.zeros_like(in_variable)
        g_noise = g_noise

        origin_InceptionResnet_model_1 = InceptionResnet_model_1(origin_variable.to(device_0)).cpu()
#         origin_InceptionResnet_model_2 = InceptionResnet_model_2(origin_variable.to(device_0)).cpu()
        origin_IR_50_model_1 = IR_50_model_1(origin_variable.to(device_0)).cpu()
#         origin_IR_152_model_1 = IR_152_model_1(origin_variable.to(device_0)).cpu()
#         origin_IR_SE_50 = IR_SE_50(origin_variable.to(device_1)).cpu()
#         origin_mobileFaceNet = mobileFaceNet(origin_variable.to(device_0)).cpu()
        origin_Insightface_iresent34 = Insightface_iresnet34(origin_variable.to(device_1)).cpu()
        origin_Insightface_iresent50 = Insightface_iresnet50(origin_variable.to(device_1)).cpu()
        origin_Insightface_iresent100 = Insightface_iresnet100(origin_variable.to(device_1)).cpu()
        origin_arcface = arc_face_ir_se_50(origin_variable.to(device_0)).cpu()
#         origin_VGG16 = VGG16.forward_GetFeature(origin_variable.to(device_1)).cpu()
        
#         origin_average_out = (origin_InceptionResnet_model_1+origin_IR_50_model_1+origin_Insightface_iresent34+\
#                origin_Insightface_iresent50+origin_Insightface_iresent100+origin_arcface)/6

#         origin_average_out  =(origin_InceptionResnet_model_1+\
#                               origin_InceptionResnet_model_2+ \
#                               origin_IR_50_model_1+\
#                               origin_IR_152_model_1+\
#                               origin_IR_SE_50+\
#                               origin_mobileFaceNet+\
#                               origin_Insightface_iresent34+\
#                               origin_Insightface_iresent50 +\
#                               origin_Insightface_iresent100 +\
#                               origin_arcface +\
#                               origin_VGG16) /11.
  
        origin_average_out  =(origin_InceptionResnet_model_1+ \
                              origin_IR_50_model_1 +\
                              origin_Insightface_iresent34+\
                              origin_Insightface_iresent50 +\
                              origin_Insightface_iresent100 +\
                              origin_arcface ) /6.

# target pix mean face
# After verification, it found no use.
#         target_mean_face_InceptionResnet_model_1 = InceptionResnet_model_1(mean_face_1.to(device_0)).cpu()
#         target_mean_face_IR_50_model_1 = IR_50_model_1(mean_face_1.to(device_0)).cpu()
#         target_mean_face_Insightface_iresent34 = Insightface_iresnet34(mean_face_1.to(device_1)).cpu()
#         target_mean_face_Insightface_iresent50 = Insightface_iresnet50(mean_face_1.to(device_1)).cpu()
#         target_mean_faceInsightface_iresent100 = Insightface_iresnet100(mean_face_1.to(device_1)).cpu()
#         target_mean_face_arcface = arc_face_ir_se_50(mean_face_1.to(device_0)).cpu()
        
#         target_mean_face_average_out = (target_mean_face_InceptionResnet_model_1 + target_mean_face_IR_50_model_1 + target_mean_face_Insightface_iresent34 + target_mean_face_Insightface_iresent50 + target_mean_faceInsightface_iresent100 + target_mean_face_arcface)/ 6

        #  sum gradient
        for i in range(steps):
            print('step: ' + str(i))
            # new_variable = input_diversity(in_variable,112,0.5)
            # 通过随机的size的padding，增加input的多样性
            mediate_InceptionResnet_model_1 = InceptionResnet_model_1(in_variable.to(device_0)).cpu()
#             mediate_InceptionResnet_model_2 = InceptionResnet_model_2(new_variable.to(device_0)).cpu()
            mediate_IR_50_model_1 = IR_50_model_1(in_variable.to(device_0)).cpu()
#             mediate_IR_152_model_1 = IR_152_model_1(new_variable.to(device_0)).cpu()
#             mediate_IR_SE_50 = IR_SE_50(new_variable.to(device_1)).cpu()
#             mediate_mobileFaceNet = mobileFaceNet(new_variable.to(device_0)).cpu()
            mediate_Insightface_iresent34 = Insightface_iresnet34(in_variable.to(device_1)).cpu()
            mediate_Insightface_iresent50 = Insightface_iresnet50(in_variable.to(device_1)).cpu()
            mediate_Insightface_iresent100 = Insightface_iresnet100(in_variable.to(device_1)).cpu()
#             mediate_VGG16 = VGG16.forward_GetFeature(new_variable.to(device_1)).cpu()
            mediate_arcface = arc_face_ir_se_50(in_variable.to(device_0)).cpu()

            # average_out = (mediate_InceptionResnet_model_1+mediate_InceptionResnet_model_2+mediate_IR_50_model_1+\
            #    mediate_IR_152_model_1+mediate_IR_SE_50+mediate_mobileFaceNet+mediate_Insightface_iresent34+\
            #    mediate_Insightface_iresent50+mediate_Insightface_iresent100+mediate_VGG16)/10
            
#             mediate_average_out = (mediate_InceptionResnet_model_1+mediate_IR_50_model_1+mediate_Insightface_iresent34+\
#                mediate_Insightface_iresent50+mediate_Insightface_iresent100+mediate_arcface)/6

#             mediate_average_out = (mediate_InceptionResnet_model_1+\
#                                    mediate_InceptionResnet_model_2+\
#                                    mediate_IR_50_model_1+\
#                                    mediate_IR_152_model_1+\
#                                    mediate_IR_SE_50+\
#                                    mediate_mobileFaceNet+\
#                                    mediate_Insightface_iresent34+\
#                                    mediate_Insightface_iresent50+\
#                                    mediate_Insightface_iresent100 +\
#                                    mediate_VGG16+\
#                                    mediate_arcface) /11.
            mediate_average_out = (mediate_InceptionResnet_model_1+ \
                                   mediate_IR_50_model_1 +\
                                   mediate_Insightface_iresent34+\
                                   mediate_Insightface_iresent50+\
                                   mediate_Insightface_iresent100 +\
                                   mediate_arcface) /6.


#             loss1 = criterion(mediate_InceptionResnet_model_1, origin_InceptionResnet_model_1) + \
#                     criterion(mediate_InceptionResnet_model_2, origin_InceptionResnet_model_2) + \
#                     criterion(mediate_IR_50_model_1, origin_IR_50_model_1) + \
#                     criterion(mediate_IR_152_model_1, origin_IR_152_model_1) + \
#                     criterion(mediate_IR_SE_50, origin_IR_SE_50) + \
#                     criterion(mediate_mobileFaceNet, origin_mobileFaceNet)+ \
#                     criterion(mediate_Insightface_iresent34, origin_Insightface_iresent34)+ \
#                     criterion(mediate_Insightface_iresent50, origin_Insightface_iresent50)  + \
#                     criterion(mediate_Insightface_iresent100, origin_Insightface_iresent100)  + \
#                     criterion(mediate_VGG16, origin_VGG16)

            loss1 = criterion(mediate_average_out, origin_average_out)


#             loss2 = criterion(mediate_InceptionResnet_model_1, mean_face) + \
#                     criterion(mediate_InceptionResnet_model_2, mean_face) + \
#                     criterion(mediate_IR_50_model_1, mean_face) + \
#                     criterion(mediate_IR_152_model_1, mean_face) +\
#                     criterion(mediate_IR_SE_50, mean_face) + \
#                     criterion(mediate_mobileFaceNet, mean_face) + \
#                     criterion(mediate_Insightface_iresent34, mean_face) + \
#                     criterion(mediate_Insightface_iresent50, mean_face) + \
#                     criterion(mediate_Insightface_iresent100, mean_face) + \
#                     criterion(mediate_VGG16, mean_face)
#             loss2 = criterion(mediate_average_out, target_mean_face_average_out)
#             loss3 = criterion(mediate_average_out,torch.zeros(512).detach())

            loss2 = criterion(mediate_average_out,mean_face)
    
            # loss3 = criterion(mediate_InceptionResnet_model_1,average_out)+ \
            #         criterion(mediate_InceptionResnet_model_2,average_out)+ \
            #         criterion(mediate_IR_50_model_1,average_out) + \
            #         criterion(mediate_IR_152_model_1,average_out) + \
            #         criterion(mediate_mobileFaceNet,average_out) + \
            #         criterion(mediate_Insightface_iresent34,average_out)+ \
            #         criterion(mediate_Insightface_iresent50,average_out) + \
            #         criterion(mediate_Insightface_iresent100,average_out)+ \
            #         criterion(mediate_VGG16,average_out)+ \
            #         criterion(mediate_IR_SE_50,average_out)


            #
            # loss = alpha * loss1 - beta* loss2 - gamma*loss3


            loss = alpha * loss1 - beta* loss2




            # print('loss : %f ' % loss,'loss1 : %f ' % loss1,'loss2 : %f ' % loss2,'loss3 : %f ' % loss3)
            # compute gradients

            loss.backward(retain_graph=True)

            g_noise = momentum * g_noise + (in_variable.grad / in_variable.grad.data.norm(1))
            g_noise = g_noise / g_noise.data.norm(1)

            g1 = g_noise
            g2 = g_noise

            # if i % 3 == 0 :
            kernel = gkern(3, 2).astype(np.float32)
            gaussian_blur1 = GaussianBlur(kernel)
            gaussian_blur1
            g1 = gaussian_blur1(g1)

            # else:
            addition = TVLoss()
            addition
            g2 = addition(g2)

            g_noise = 0.25 * g1 + 0.75 * g2


            in_variable.data = in_variable.data + (
                        (eps / 255.) * torch.sign(g_noise))  # * torch.from_numpy(mat).unsqueeze(0).float()

            in_variable.data = clip_by_tensor(in_variable.data,in_variable_min.data,in_variable_max.data)


    
            in_variable.grad.data.zero_()  # unnecessary

#             del new_variable

            # g_noise = in_variable.data - origin_variable
            # g_noise.clamp_(-0.2, 0.2)
            # in_variable.data = origin_variable + g_noise

        # deprocess image
        for i in range(len(in_variable.data.cpu().numpy())):
            adv = in_variable.data.cpu().numpy()[i]  # (3, 112, 112)
            perturbation = (adv - in_tensor.cpu().numpy())

            adv = adv * 128.0 + 127.0
            adv = adv.swapaxes(0, 1).swapaxes(1, 2)
            adv = adv[..., ::-1]
            adv = np.clip(adv, 0, 255).astype(np.uint8)

            # sample_dir = './target_mean_face/'
            # if not os.path.exists(sample_dir):
            #     os.makedirs(sample_dir)

            advimg = sample_dir + path[i].split('/')[-1].split('.')[-2] + '.jpg'
            print(advimg)
            cv2.imwrite(advimg, adv)
            print("save path is " + advimg)
            print('cost time is %.2f s ' % (time.time() - start))

        count += 6
        progressRate = count / 712.


if __name__ == '__main__':
    main()


