import torch
import torch.nn.functional as TNF
import torchvision.transforms.functional as TF
from network import ResidualBlock, ResNet
from hyperparameters import *
import os
from PIL import Image


def test(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')# 检测device是否支持cuda

    im = Image.open(img)
    im = im.resize((32, 32))
    # im.save("result.jpg")
    tensor_test = TF.to_tensor(im)# 此时的im是一张32*32的PIL.Image.Image，这一步是将图片转换为tensor

    path_model = os.path.join("model", "resnet.ckpt")
    model = ResNet(ResidualBlock, layers, number_classes).to(device)# 初始化ResNet
    model.load_state_dict(torch.load(path_model))# 装载已经训练好的ResNet模型
    # print(model)
    model.eval()# 开始评估

    with torch.no_grad():
        list_test = tensor_test.tolist()# 将tensor转化为list
        list_test = [list_test]
        tensor_test = torch.Tensor(list_test)
        images = tensor_test.to(device)# 把执行设备从cpu改为gpu
        outputs = model(images)# 将这张图送进model里面测试
        outputs = TNF.softmax(outputs, dim=1)
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted.data)

    if predicted[0] == 0:# 是人脸
        return 0
    else:
        return 1# 不是人脸


if __name__ == "__main__":
    # print(test("test.jpg"))

    # P_1 = os.path.join("data", "dataset")
    # P_2 = os.path.join(P_1, "face")
    # for n in os.listdir(P_2):
    #     path_file = os.path.join(P_2, n)
    #     print(test(path_file))

    P_1 = os.path.join("data", "dataset")
    P_2 = os.path.join(P_1, "others")
    for n in os.listdir(P_2):
        path_file = os.path.join(P_2, n)# path_file是others数据集里面的图片的绝对路径
        print(test(path_file))
