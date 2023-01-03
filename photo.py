# -*- coding: utf-8 -*-
# @Time    : 2022/12/22 11:14
# @Author  : Hush
# @Email   : crush@tju.edu.cn


import glob
from PIL import Image
import os
import zipfile

# 获取image文件夹下的图片路径
path = "/Users/crush/Downloads/seeprettyface_asian_stars/"

# 获取到当前文件的目录，并检查是否有result文件夹，如果没有则创建
# image、result文件夹及该python文件路径均为：D:\PythonSpace\
Newimg_Path = "/Users/crush/Downloads/stars/"
if not os.path.exists(Newimg_Path):
    os.makedirs(Newimg_Path)
def changesize(img):
    img_name = img
    # 打开图片
    oldimg = Image.open(path + img_name)
    # 大小缩放为64*64
    new_img = oldimg.resize((64, 64))
    # 以原名称存储图片
    new_img.save(Newimg_Path + img_name)
# 读取图片及名称
for img in os.listdir(path):
    # 不同格式
    if (img.endswith('.gif') or img.endswith('.png') or img.endswith('.jpg')):
        # 修改图片，存储图片
        changesize(img)

