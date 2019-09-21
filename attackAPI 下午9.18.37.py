#python3
# -*- coding: utf-8 -*-



import hashlib
import time
import random
import string
import requests
import base64
import requests
import cv2
import numpy as np
from urllib.parse import urlencode
import json #用于post后得到的字符串到字典的转换
import os

app_id = '2121645568'
app_key = '2rSUYrEdQbLBgQiP'


#创建一个个体（Person）
def create_person(person_id,person_name,img,group_ids):
    #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
    time_stamp=str(int(time.time()))
    #请求随机字符串，用于保证签名不可预测,16代表16位
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    params = {'app_id':app_id,                #请求包，需要根据不同的任务修改，基本相同
              'image':img,                    #文字类的任务可能是‘text’，由主函数传递进来
              'time_stamp':time_stamp,        #时间戳，都一样
              'nonce_str':nonce_str,          #随机字符串，都一样
              'group_ids':group_ids,
              'person_id':person_id,
              'person_name':person_name
              #'sign':''                      #签名不参与鉴权计算，只是列出来示意
             }

    sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
    sort_dict.append(('app_key',app_key))   #尾部添加appkey
    rawtext= urlencode(sort_dict).encode()  #urlencod编码
    sha = hashlib.md5()
    sha.update(rawtext)
    md5text= sha.hexdigest().upper()        #MD5加密计算
    params['sign']=md5text                  #将签名赋值到sign

    url = "https://api.ai.qq.com/fcgi-bin/face/face_newperson"  # 人脸分析
    res = requests.post(url,params).json()
    print(res)
    return res
#删除一个个体（Person）
def deleta_person(person_id):
    #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
    time_stamp=str(int(time.time()))
    #请求随机字符串，用于保证签名不可预测,16代表16位
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    params = {'app_id':app_id,
              'time_stamp':time_stamp,
              'nonce_str':nonce_str,
              'person_id':person_id,
              #'sign':''                      #签名不参与鉴权计算，只是列出来示意
             }

    sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
    sort_dict.append(('app_key',app_key))   #尾部添加appkey
    rawtext= urlencode(sort_dict).encode()  #urlencod编码
    sha = hashlib.md5()
    sha.update(rawtext)
    md5text= sha.hexdigest().upper()        #MD5加密计算
    params['sign']=md5text                  #将签名赋值到sign

    url = "https://api.ai.qq.com/fcgi-bin/face/face_delperson"  # 人脸分析
    try:
        res = requests.post(url,params).json()
    except Exception:
        print(Exception)

    print(res)
    return res
#获取应用下所有的组（Group）ID列表
def getgroupids():
    #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
    time_stamp=str(int(time.time()))
    #请求随机字符串，用于保证签名不可预测,16代表16位
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    params = {'app_id':app_id,
              'time_stamp':time_stamp,        #时间戳，都一样
              'nonce_str':nonce_str,
              #'sign':''                      #签名不参与鉴权计算，只是列出来示意
             }

    sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
    sort_dict.append(('app_key',app_key))   #尾部添加appkey
    rawtext= urlencode(sort_dict).encode()  #urlencod编码
    sha = hashlib.md5()
    sha.update(rawtext)
    md5text= sha.hexdigest().upper()        #MD5加密计算
    params['sign']=md5text                  #将签名赋值到sign

    url = "https://api.ai.qq.com/fcgi-bin/face/face_getgroupids"
    res = requests.post(url,params).json()
    print(res)

# def addface2person(person_id,img):
#     #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
#     time_stamp=str(int(time.time()))
#     #请求随机字符串，用于保证签名不可预测,16代表16位
#     nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))
#
#     params = {'app_id':app_id,                #请求包，需要根据不同的任务修改，基本相同
#               'image':img,                    #文字类的任务可能是‘text’，由主函数传递进来
#               'time_stamp':time_stamp,        #时间戳，都一样
#               'nonce_str':nonce_str,
#               'person_id':person_id,
#               #'sign':''                      #签名不参与鉴权计算，只是列出来示意
#              }
#
#     sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
#     sort_dict.append(('app_key',app_key))   #尾部添加appkey
#     rawtext= urlencode(sort_dict).encode()  #urlencod编码
#     sha = hashlib.md5()
#     sha.update(rawtext)
#     md5text= sha.hexdigest().upper()        #MD5加密计算
#     params['sign']=md5text                  #将签名赋值到sign
#
#     url = "https://api.ai.qq.com/fcgi-bin/face/face_addface"
#     res = requests.post(url,params).json()
#     print(res)
#根据个体（Person）ID 获取人脸（Face）ID列表
def face_getfaceids(person_id):
    #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
    time_stamp=str(int(time.time()))
    #请求随机字符串，用于保证签名不可预测,16代表16位
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    params = {'app_id':app_id,
              'time_stamp':time_stamp,        #时间戳，都一样
              'nonce_str':nonce_str,
              'person_id':person_id
              #'sign':''                      #签名不参与鉴权计算，只是列出来示意
             }

    sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
    sort_dict.append(('app_key',app_key))   #尾部添加appkey
    rawtext= urlencode(sort_dict).encode()  #urlencod编码
    sha = hashlib.md5()
    sha.update(rawtext)
    md5text= sha.hexdigest().upper()        #MD5加密计算
    params['sign']=md5text                  #将签名赋值到sign

    url = "https://api.ai.qq.com/fcgi-bin/face/face_getfaceids"
    res = requests.post(url,params).json()
    print(res)


def face_getpersonids(group_id):
    #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
    time_stamp=str(int(time.time()))
    #请求随机字符串，用于保证签名不可预测,16代表16位
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    params = {'app_id':app_id,
              'time_stamp':time_stamp,        #时间戳，都一样
              'nonce_str':nonce_str,
              'group_id':group_id
              #'sign':''                      #签名不参与鉴权计算，只是列出来示意
             }

    sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
    sort_dict.append(('app_key',app_key))   #尾部添加appkey
    rawtext= urlencode(sort_dict).encode()  #urlencod编码
    sha = hashlib.md5()
    sha.update(rawtext)
    md5text= sha.hexdigest().upper()        #MD5加密计算
    params['sign']=md5text                  #将签名赋值到sign

    url = "https://api.ai.qq.com/fcgi-bin/face/face_getpersonids"
    res = requests.post(url,params).json()
    print(res)
    return res

#对请求图片中的人脸进行搜索
def face_faceidentify(group_id,img,topn):
    #请求时间戳（秒级），用于防止请求重放（保证签名5分钟有效
    time_stamp=str(int(time.time()))
    #请求随机字符串，用于保证签名不可预测,16代表16位
    nonce_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))

    params = {'app_id':app_id,                #请求包，需要根据不同的任务修改，基本相同
              'image':img,                    #文字类的任务可能是‘text’，由主函数传递进来
              'time_stamp':time_stamp,        #时间戳，都一样
              'nonce_str':nonce_str,          #随机字符串，都一样
              'group_id':group_id,
              'topn':int(topn)
             }

    sort_dict= sorted(params.items(), key=lambda item:item[0], reverse = False)  #字典排序
    sort_dict.append(('app_key',app_key))   #尾部添加appkey
    rawtext= urlencode(sort_dict).encode()  #urlencod编码
    sha = hashlib.md5()
    sha.update(rawtext)
    md5text= sha.hexdigest().upper()        #MD5加密计算
    params['sign']=md5text                  #将签名赋值到sign

    url = "https://api.ai.qq.com/fcgi-bin/face/face_faceidentify"
    res = requests.post(url,params).json()
    print(res)

def process_img(pic_path):
    frame=cv2.imread(pic_path)
    nparry_encode = cv2.imencode('.jpg', frame)[1]
    data_encode = np.array(nparry_encode)
    img = base64.b64encode(data_encode)
    return img

def add_lfw_to_tencent():
    paths = []
    picpath = '/notebooks/Workspace/tmp/pycharm_project_314/TianChi/images'
    for root, dirs, files in os.walk(picpath):
        for f in files:
            paths.append(os.path.join(root, f))
    for p in paths:
        img = process_img(p)
        persion_id = p.split('/')[-1].split('.')[0]
        persion_name = persion_id
        # res = deleta_person(persion_id)
        # time.sleep(3)

        # if res['msg'] == 'ok':
        #     continue
        # else:
        #     time.sleep(1)
        # while 1:
        #     if face_getfaceids(persion_id) or create_person(persion_id,persion_name,img,'lfw-712')['data']['msg'] == 'ok':
        #         break
        create_person(persion_id, persion_name, img, 'lfw-712')
        time.sleep(5)
if __name__ == '__main__':

    #
    # getgroupids()
    # face_faceidentify('lfw-712',process_img('../images/00001.jpg'),1)
    # face_getfaceids('00001')
    # deleta_person('00001')
    # face_getpersonids('lfw-712')
    # persion = face_getpersonids('lfw-712')['data']['person_ids']
    # for p in persion:
    #     deleta_person(persion)

    add_lfw_to_tencent()
    # face_getfaceids('00001')