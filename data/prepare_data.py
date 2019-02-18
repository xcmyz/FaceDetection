# I change the path of the dataset because in Visual Studio, only one execution path is acceptable. So I set the execution path as Face_Binary_Classfication folder. Therefore, I change the string "lfw_funneled" to "data\\lfw_funneled", "cifar-100-python" is same.
# When you execute the python script, if you execute it directly in command line, change the path string before you execute it.
import cv2 as cv
import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL
import zipfile


def prepareFaceData():
    files = os.listdir("lfw_funneled")
    # print(files)
    # test  = 0
    cnt = 0
    for file in files:
        if os.path.isdir(os.path.join("lfw_funneled", file)):# lfw数据集的人脸图像均在文件夹下，因此先判断有无该文件夹
            # print("*****")
            p = os.path.join("lfw_funneled", file)
            # print(p)
            files_1 = os.listdir(p)# 获得其中的图片的名字列表，files_1是一个包含该文件夹下所有文件名称的list
            for file_1 in files_1:
                if not os.path.isdir(os.path.join(p, file_1)):# 判断没有子文件夹了
                    pic_path = os.path.join(p, file_1)# 获得图片的绝对路径（基于运行路径下的绝对路径）
                    im = cv.imread(pic_path)
                    # test = im
                    # break
                    # print(pic_path)
                    # cv.imshow("test", im)
                    # print("#####")
                    im = cv.resize(im, (32, 32))# 化为32*32大小的图片
                    path_pic = os.path.join("dataset", "face")
                    num = cnt
                    cnt = cnt + 1
                    name = str(num) + ".jpg"
                    N = os.path.join(path_pic, name)# 生成路径dataset\\face\\xxx.jpg
                    cv.imwrite(N, im)

                    if cnt % 1000 == 0:# 每一千张生成输出一个信息
                        print("Done %d" % cnt)
            # break
    # cv.imshow("test", test)
    # cv.waitKey()


def prepareOtherData():
    #file_list=os.listdir(r'.')

    #for file_name in file_list:
    #    if os.path.splitext(file_name)[1]=='.zip':
    #        file_zip=zipfile.ZipFile(file_name,'r')
    #        for file in file_zip.namelist():
    #            file_zip.extract(file,r'coco')
    #        file_zip.close()
    files = os.listdir("coco")
    cnt = 0
    for file in files:
        if os.path.isdir(os.path.join("coco", file)):# coco数据集的人脸图像均在文件夹下，因此先判断有无该文件夹
            p = os.path.join("coco", file)
        files_1 = os.listdir(p)# 获得其中的图片的名字列表，files_1是一个包含该文件夹下所有文件名称的list
        for file_1 in files_1:
            if not os.path.isdir(os.path.join(p, file_1)):# 判断没有子文件夹了
                pic_path = os.path.join(p, file_1)# 获得图片的绝对路径（基于运行路径下的绝对路径）
            im = cv.imread(pic_path)
            im = cv.resize(im, (100, 100))# 化为100*100大小的图片
            path_pic = os.path.join("dataset", "others")
            num = cnt
            cnt = cnt + 1
            name = str(num) + ".jpg"
            N = os.path.join(path_pic, name)# 生成路径dataset\\others\\xxx.jpg
            cv.imwrite(N, im)

            if cnt % 1000 == 0:# 每一千张生成输出一个信息
                print("Done %d" % cnt)


if __name__ == "__main__":

    # prepareFaceData()
    prepareOtherData()
    # flip("dataset/others/9.jpg")
