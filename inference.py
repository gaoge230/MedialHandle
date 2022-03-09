#!/usr/bin/env/ python
# -*- coding: utf-8 -*-
# @File  : inference.py 
# @Author: 高歌
# @Date  : 2021/1/2
# @Desc  :

import argparse
import os
import torch
import numpy as np
import cv2
from model import UNet, ResUnet, XUnet, Mynet, Mynet1, Mynet2, Mynet3
import torch
from PIL import Image
from torchvision import transforms

def view_result():
    # input_img_path = args.input_img_path
    # model_path = args.model_path
    # save = args.view_path

    input_img_path = './data/LoadImage/result9_img.png'
    # input_img_path = './view/valid/images/0.png'
    model_path = 'data/model/cardia_sstu_net_train_best_model.pth'
    save = './data/result/'
    # normalize = transforms.Normalize(mean=[0.156, 0.156, 0.156],
    #                                 std=[0.174, 0.174, 0.174])  # 不同的数据集需要自己计算。 head数据集
    normalize = transforms.Normalize(mean=[0.247, 0.260, 0.287],
                                     std=[0.189, 0.171, 0.121])  # 不同的数据集需要自己计算。   cardia_all数据集
    # normalize = transforms.Normalize(mean=[0.191, 0.191, 0.191],
    #                                 std=[0.218, 0.218, 0.218])  # 不同的数据集需要自己计算。   camus_256
    # transforms
    transform = transforms.Compose(
        [   #transforms.ToPILImage(),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomGrayscale(p=0.2),  # 依概率p转为灰度图
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),  # 修改修改亮度、对比度和饱和度
            transforms.ToTensor(),
            normalize
         ])
    device = torch.device('cuda')
    model = UNet(3,1)
    #model = ResUnet(3,1)
    #model = AttU_Net(3,1)
    #model = NestedUNet(3,1)

    pretrained_dict = torch.load(model_path)

    #print(pretrained_dict)

    model.load_state_dict(pretrained_dict)
    model.to(device)
    scale = (224, 224)
    img_o = Image.open(input_img_path).resize(scale)
    #img_o.show()

    img = transform(img_o).unsqueeze(0)

    print("img_np:",img.shape)
    #img_rensor= torch.from_numpy(img/255.0)

    #img = img_rensor.unsqueeze(0).unsqueeze(0).float()
    print("img:", img.size())
    img_ = img.to(device)
    with torch.no_grad():
        outputs = model(img_)
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    output_np = outputs.cpu().numpy().squeeze()
    print("output_np:", output_np.shape)
    name  = os.path.basename(input_img_path).split('.')[0]
    print(os.path.dirname(os.path.dirname(input_img_path)))
    #target= Image.open(os.path.dirname(os.path.dirname(input_img_path))+f'/mask/{name}_label.png').resize(scale)

    output = Image.fromarray((output_np*255).astype('uint8')).convert('1')
    #output.show()
    img_o.save(save + f'/{name}.png')
    output.save(save + f'/{name}_result.png')
    #target.save(save + f'/{name}_target.png')
    return save + f'/{name}_result.png'

def view_result1(loadpath,modelPath=None,modelType=-1,savePath=None):
    # input_img_path = args.input_img_path
    # model_path = args.model_path
    # save = args.view_path

    input_img_path = loadpath
    # input_img_path = './view/valid/images/0.png'
    model_path = 'data/model/cardia_sstu_net_train_best_model.pth'
    if modelPath!=None:
        model_path=modelPath

    save = './data/result/'
    if savePath!=None:
        save = savePath

    normalize = transforms.Normalize(mean=[0.247, 0.260, 0.287],
                                     std=[0.189, 0.171, 0.121])  # 不同的数据集需要自己计算。   cardia_all数据集

    # transforms
    transform = transforms.Compose(
        [   #transforms.ToPILImage(),
            #transforms.RandomHorizontalFlip(p=0.5),
            #transforms.RandomVerticalFlip(p=0.5),
            #transforms.RandomGrayscale(p=0.2),  # 依概率p转为灰度图
            #transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),  # 修改修改亮度、对比度和饱和度
            transforms.ToTensor(),
            normalize
         ])
    device = torch.device('cpu')

    model = None
    if modelType==0:
        model = ResUnet(3,1)
    elif modelType==1:
        model = ResUnet(3,1) #
    else:
        model = UNet(3,1)

    #model = NestedUNet(3,1)
    pretrained_dict = torch.load(model_path,map_location='cpu')
    model.load_state_dict(pretrained_dict)
    model.to(device)
    scale = (224, 224)
    img_o = Image.open(input_img_path).resize(scale)
    img = transform(img_o).unsqueeze(0)
    print("img_np:",img.shape)
    print("img:", img.size())
    img_ = img.to(device)
    with torch.no_grad():
        outputs = model(img_)
    outputs = torch.sigmoid(outputs)
    outputs = (outputs > 0.5).float()
    output_np = outputs.cpu().numpy().squeeze()
    print("output_np:", output_np.shape)
    name  = os.path.basename(input_img_path).split('.')[0]
    print(os.path.dirname(os.path.dirname(input_img_path)))
    #target= Image.open(os.path.dirname(os.path.dirname(input_img_path))+f'/mask/{name}_label.png').resize(scale)

    output = Image.fromarray((output_np*255).astype('uint8')).convert('1')
    #output.show()
    img_o.save(save + f'/{name}.png')
    output.save(save + f'/{name}_result.png')
    #target.save(save + f'/{name}_target.png')
    return save + f'/{name}_result.png'

if __name__ == "__main__":
   # # parser = argparse.ArgumentParser()
   #  parser.add_argument('--model_path', type=str, default='./view/cardia_unet_train_best_model.pth') # 训练模型保存目录
   #  parser.add_argument('--input_img_path', type=str, default='./view/valid/images/0.png') # 图片
   #  parser.add_argument('--view_path', type=str, default='./view/result')
   #  args = parser.parse_args()
   #  print(args)
   #  view_result()
    view_result1("./image/result0_img.png")