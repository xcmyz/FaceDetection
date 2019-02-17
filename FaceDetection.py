# encoding:utf-8
import urllib.request
import base64

import cv2 as cv
import math


def drawOffsetRec(im, leftupCoor, rightdownCoor, color, offset, thickness=1):
    # offset is °

    k = 1.0/math.tan(math.pi * ((90.0 - offset)/180.0))# 获得倾角对应的斜率，k=1/tan\alpha

	# 啥意思？
    x_ld = (((k * leftupCoor[0] - leftupCoor[1]) +
             (rightdownCoor[0]/k + rightdownCoor[1])))/(k + 1.0/k)
    y_ld = k * x_ld - k * leftupCoor[0] + leftupCoor[1]

    x_ru = ((leftupCoor[0]/k + leftupCoor[1]) +
            (k * rightdownCoor[0] - rightdownCoor[1]))/(k + 1.0/k)
    y_ru = k * (x_ru - rightdownCoor[0]) + rightdownCoor[1]

    x_ld = int(x_ld)
    y_ld = int(y_ld)
    x_ru = int(x_ru)
    y_ru = int(y_ru)

    # print(type(leftupCoor))
    # print(type(rightdownCoor))
    # print(type(color))

    cv.line(im, leftupCoor, (x_ld, y_ld), color, thickness)
    cv.line(im, leftupCoor, (x_ru, y_ru), color, thickness)
    cv.line(im, rightdownCoor, (x_ru, y_ru), color, thickness)
    cv.line(im, rightdownCoor, (x_ld, y_ld), color, thickness)

    return im


def faceDetection(file_name):
	# 利用AK和SK给百度服务器发送请求，得到初始化上下文context
    # client_id 为官网获取的AK， client_secret 为官网获取的SK
    host = 'https://aip.baidubce.com/oauth/2.0/' + \
        'token?grant_type=client_credentials&client_id=' + \
        'RHRpebbdKKyeMwGbrgFgZobB&client_secret=mjtUvxV7p3Ou9tk8rafKBgnQfuHteN6S'
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    content = response.read()
    content = content.decode("utf-8")
    content = eval(content)

    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"

    im = cv.imread(file_name)
    with open(file_name, "rb")as f:
        ls_f = base64.b64encode(f.read())# 使用base64对图片进行编码，百度技术文档说明需要传输base64类型的数据给服务器

    image = ls_f
    params = {"image": image, "image_type": "BASE64",
              "face_field": "faceshape,facetype", "max_face_num": 10}
    params = urllib.parse.urlencode(params).encode('utf-8')# utf-8形式对params编码，便于流式传输

    access_token = content["access_token"]# 获得token
    request_url = request_url + "?access_token=" + access_token# 获得基于token的url
    request = urllib.request.Request(url=request_url, data=params)# 初始化request
    request.add_header('Content-Type', 'application/json')# 加上header
    response = urllib.request.urlopen(request)# 打开url，获得基于HTTP的回应
    content_ = response.read()
    content_ = content_.decode("utf-8")# 对服务器发送回来的数据进行解码
    if (content_.find("null")) == -1:# 没有任何空串，网络没有问题
        content_ = eval(content_)# 解码成一个dict
        if content_:
            if False:
                print(content_)
    else:
        print("ERROR!")
        return im

    # draw rec
    for index in range(content_["result"]["face_num"]): # 得出content表达式中人脸的个数

        # print(content_["result"]["face_list"][index]["location"])
        process_dict = content_["result"]["face_list"][index]["location"] # 该表达式返回左上角坐标，长度，宽度以及旋转角度

        offset = 0.0
        if process_dict["rotation"] >= 0 and process_dict["rotation"] <= 90: #判断旋转角度在什么范围，以决定是顺时针旋转还是逆时针旋转
            offset = process_dict["rotation"] + 0.0
        elif process_dict["rotation"] <= 0 and process_dict["rotation"] >= -90:
            offset = process_dict["rotation"] + 0.0
        else:
            if process_dict["rotation"] > 90:
                offset = process_dict["rotation"] - 90
                offset = offset
            else:
                offset = process_dict["rotation"] + 90
                offset = offset

        # print(offset)
        if process_dict["rotation"] < -90 or process_dict["rotation"] > 90: #判断rotation是否为大偏角，（为啥要减？）
            x_lu = int(process_dict["left"])
            y_lu = int(process_dict["top"])

            x_rd = x_lu - process_dict["width"]
            y_rd = y_lu - process_dict["height"]
            x_rd = int(x_rd)
            y_rd = int(y_rd)
        else:
            x_lu = int(process_dict["left"]) # lu，left-up，左上角
            y_lu = int(process_dict["top"])

            x_rd = x_lu + process_dict["width"] # rd，right-down，右下角
            y_rd = y_lu + process_dict["height"]
            x_rd = int(x_rd)
            y_rd = int(y_rd)

        im = drawOffsetRec(im, (x_lu, y_lu), (x_rd, y_rd), (0, 0, 255), offset)
        cnt = index # cnt的实际作用是什么
        cnt = str(cnt)
        # cv.imshow(cnt, im)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    # cv.imshow("test", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    return im


if __name__ == "__main__":

    # im = faceDetection("test_1.jpg")
    # cv.imshow("test_1", im)

    im = faceDetection("FRAME_CUT_EQU\\13_1.jpg")
    cv.imshow("test_2", im)

    # im = faceDetection("test_3.jpg")
    # cv.imshow("test_3", im)

    cv.waitKey(0)
    cv.destroyAllWindows()