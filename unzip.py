# COCO dataset unzip and resize code
import zipfile
import os
import cv2 as cv

#file_list=os.listdir(r'.')

#for file_name in file_list:
#	if os.path.splitext(file_name)[1]=='.zip':
#		file_zip=zipfile.ZipFile(file_name,'r')
#		for file in file_zip.namelist():
#			file_zip.extract(file,r'.\\data\\coco')
#		file_zip.close()
files = os.listdir(".\\data\\coco")
cnt = 0
for file in files:
	if os.path.isdir(os.path.join(".\\data\\coco", file)):# coco数据集的人脸图像均在文件夹下，因此先判断有无该文件夹
		p = os.path.join(".\\data\\coco", file)
	files_1 = os.listdir(p)# 获得其中的图片的名字列表，files_1是一个包含该文件夹下所有文件名称的list
	for file_1 in files_1:
		if not os.path.isdir(os.path.join(p, file_1)):# 判断没有子文件夹了
			pic_path = os.path.join(p, file_1)# 获得图片的绝对路径（基于运行路径下的绝对路径）
		im = cv.imread(pic_path)
		im = cv.resize(im, (32, 32))# 化为32*32大小的图片
		cnt = cnt + 1
		cv.imwrite(pic_path, im)

		if cnt % 1000 == 0:# 每一千张生成输出一个信息
			print("Done %d" % cnt)
