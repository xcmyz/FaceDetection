import requests
import json
import base64
import cv2
import os

def get_landmark(file_name):
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/' + \
               'token?grant_type=client_credentials&client_id=' + \
               'RHRpebbdKKyeMwGbrgFgZobB&client_secret=mjtUvxV7p3Ou9tk8rafKBgnQfuHteN6S'
    
    r = requests.get(host) # 得到返回的带有access_token的字符串
    r = json.loads(r.text) # 将字符串转换为字典
    access_token = r['access_token']

    with open(file_name, "rb")as f:
            ls_f = base64.b64encode(f.read())  # 使用base64对图片进行编码，百度技术文档说明需要传输base64类型的数据给服务器
    image = ls_f
    post = 'https://aip.baidubce.com/rest/2.0/face/v3/detect'
    kv = {'image': image, 'image_type': 'BASE64', 'face_field': 'landmark', 'max_face_num': 1, 'access_token': access_token }
    # data表单的参数列表

    r = requests.post(post, data=kv,headers={'Content_Type':'application/json'}) # 发送post请求，向百度提交以上工单

    # print(r.status_code) # 看看连接码
    result=json.loads(r.text)
    coor=result['result']['face_list'][0]['landmark'] # 4个坐标组成的数组，元素是字典
    print(coor)
    return coor

if __name__ == '__main__':
    get_landmark('D:/桌面/dachuang/test.jpg')
