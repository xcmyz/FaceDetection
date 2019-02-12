import torch
import torch.nn.functional as TNF
import torchvision.transforms.functional as TF
from network import ResidualBlock, ResNet
from hyperparameters import *
import os
from PIL import Image


def test(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    im = Image.open(img)
    im = im.resize((32, 32))
    # im.save("result.jpg")
    tensor_test = TF.to_tensor(im)

    path_model = os.path.join("model", "resnet.ckpt")
    model = ResNet(ResidualBlock, layers, number_classes).to(device)
    model.load_state_dict(torch.load(path_model))
    # print(model)
    model.eval()

    with torch.no_grad():
        list_test = tensor_test.tolist()
        list_test = [list_test]
        tensor_test = torch.Tensor(list_test)
        images = tensor_test.to(device)
        outputs = model(images)
        outputs = TNF.softmax(outputs, dim=1)
        # print(outputs)
        _, predicted = torch.max(outputs.data, 1)
        # print(predicted.data)

    if predicted[0] == 0:
        return 0
    else:
        return 1


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
        path_file = os.path.join(P_2, n)
        print(test(path_file))
