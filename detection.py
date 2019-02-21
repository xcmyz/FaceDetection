#encoding:utf-8
# This code is for the frame's face detection and generate standard human face data
import urllib.request
import base64
import cv2
import math
import os,sys,time

face_cnt=0
print('Baidu API request initializing...')
# 百度服务器请求初始化
host = 'https://aip.baidubce.com/oauth/2.0/' + \
        'token?grant_type=client_credentials&client_id=' + \
        'RHRpebbdKKyeMwGbrgFgZobB&client_secret=mjtUvxV7p3Ou9tk8rafKBgnQfuHteN6S'
request=urllib.request.Request(host)
request.add_header('Content-Type','application/json; charset=UTF-8')
response=urllib.request.urlopen(request)
content=response.read()
content=content.decode('utf-8')
content=eval(content)

request_url='https://aip.baidubce.com/rest/2.0/face/v3/detect'
print('Baidu API request initialize finished.')
# 加载图片list
files=os.listdir('FRAME_CUT_EQU')
print('Photo path loading finished! Processing...')
flag=0
for file in files:
	if file=='422_0.jpg':# 第一张检测不到人脸的糊图
		flag=1
		face_cnt=6001
	if flag==0:
		continue
	request_url='https://aip.baidubce.com/rest/2.0/face/v3/detect'
	path=os.path.join('FRAME_CUT_EQU',file)
	if os.path.isdir(path):
		print('File format error',file=sys.stderr)
		os.system('pause')
		os._exit(1)
	img=cv2.imread(path)
	if img is None:
		print('Loading image error, please check the data.')
		os.system('pause')
		os._exit(1)

# 送到服务器进行识别
	with open(path,'rb') as f:
		imb64=base64.b64encode(f.read())

	image = imb64
	params = {'image': image, 'image_type': 'BASE64',
              'face_field': 'faceshape,facetype', 'max_face_num': 10}
	params = urllib.parse.urlencode(params).encode('utf-8')# utf-8形式对params编码，便于流式传输

	access_token = content['access_token']# 获得token
	request_url = request_url + '?access_token=' + access_token# 获得基于token的url
	request = urllib.request.Request(url=request_url, data=params)# 初始化request
	request.add_header('Content-Type', 'application/json')# 加上header
	response = urllib.request.urlopen(request)# 打开url，获得基于HTTP的回应
	content_ = response.read()
	content_ = content_.decode('utf-8')# 对服务器发送回来的数据进行解码
	if (content_.find('null')) == -1:# 没有任何空串，结果没有问题
		content_ = eval(content_)# 解码成一个dict
		#if content_:
		#	if False:
		#		print(content_)
		if content_['error_msg'] is not 'SUCCESS':
			print('Baidu\'s face detection failed.')
			print('The error message is : %s' % content_['error_msg'])
			continue
	else:
		print('ERROR!',file=sys.stderr)
		print('Error message : %s' % content_)
		print('The error picture\'s name is %s' % file)
		continue

	# 对发下来的包进行分析
	# 判断执行是否成功
	for index in range(content_['result']['face_num']):
		# 概率阈值为0.85
		if content_['result']['face_list'][index]['face_probability'] - 0.85 < 1e-5:
			continue
		# 获得人脸相关参数
		process_dict=content_['result']['face_list'][index]['location']
		# TODO: 利用数学推导出读取斜框数据的方式，已知矩形的左上角顶点坐标，长，宽，以及与水平轴所成的角度
		# 首先 没有旋转 如果得到的是空矩阵，就不写入
		face=img[int(process_dict['top']*0.85):int(process_dict['top']+process_dict['height']),int(process_dict['left']):int(process_dict['left']+process_dict['width'])]
		if face.size==0:
			continue
		face=cv2.resize(face,(32,32))
		if not os.path.isdir('classmate_face2'): #not exist
			os.mkdir('classmate_face2')
		face_name=str(face_cnt)+'.jpg'
		face_cnt = face_cnt + 1
		path=os.path.join('classmate_face2',face_name)
		cv2.imwrite(path,face)
		if (face_cnt) % 1000 == 0:
			print('Done',face_cnt)
	time.sleep(0.5)# 防止超过服务器每秒响应请求

print('Face process complete, total amount:',face_cnt)