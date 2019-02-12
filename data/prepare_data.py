import cv2 as cv
import os
import pickle
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL


def prepareFaceData():
    files = os.listdir("lfw_funneled")
    # print(files)
    # test  = 0
    cnt = 0
    for file in files:
        if os.path.isdir(os.path.join("lfw_funneled", file)):
            # print("*****")
            p = os.path.join("lfw_funneled", file)
            # print(p)
            files_1 = os.listdir(p)
            for file_1 in files_1:
                if not os.path.isdir(os.path.join(p, file_1)):
                    pic_path = os.path.join(p, file_1)
                    im = cv.imread(pic_path)
                    # test = im
                    # break
                    # print(pic_path)
                    # cv.imshow("test", im)
                    # print("#####")
                    im = cv.resize(im, (32, 32))
                    path_pic = os.path.join("dataset", "face")
                    num = cnt
                    cnt = cnt + 1
                    name = str(num) + ".jpg"
                    N = os.path.join(path_pic, name)
                    cv.imwrite(N, im)

                    if cnt % 1000 == 0:
                        print("Done %d" % cnt)
            # break
    # cv.imshow("test", test)
    # cv.waitKey()


def prepareOtherData():
    # file_name= "cifar-100-python.tar"

    # def unpickle(file):
    #     with open(file, 'rb') as fo:
    #         dict = pickle.load(fo, encoding='bytes')
    #     return dict

    # print(unpickle(file_name))

    # transform = transforms.Compose([
    # transforms.Pad(4),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(32),
    # transforms.ToTensor()])

    # train_dataset = torchvision.datasets.CIFAR100(
    #     root='', train=True, transform=transforms.ToTensor())

    # train_dataset = torchvision.datasets.CIFAR100(
    #     root='', train=True, transform=None)

    # test_d = []
    # for i , d in enumerate(train_dataset):
    #     # print(d)
    #     if d[1] == 11:
    #         test_d.append(d[0])
    #         # break
    # plt.figure()
    # plt.imshow(test_d[3])
    # plt.show()

    # train_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset, batch_size=1, shuffle=True)

    # sampler = torch.utils.data.SequentialSampler(train_dataset)
    # print(list(torch.utils.data.BatchSampler(sampler, batch_size=1, drop_last=False)))

    # print(train_dataset)
    # print(sample)

    # print(train_loader)
    # for i, (images, labels) in enumerate(train_loader):
    #     # print((images, labels))
    #     # print(len(images))
    #     print(len(images[0]))
    #     # print(images[0][1][0])
    #     # print(len(images[0][1]))
    #     # print(images[0])

    # for _, something in enumerate(train_loader):
    #     print(something)
    #     # transforms.ToPILImage

    # # cifar100_data = np.load(os.path.join("cifar-100-python", "train"))
    # cifar100_data = pickle.load(os.path.join("cifar-100-python", "train"))
    # print(cifar100_data)

    train_dataset = torchvision.datasets.CIFAR100(
        root='', train=True, transform=None)

    cifar100_class = np.load(os.path.join("cifar-100-python", "meta"))
    # print(cifar100_data)
    list_delete = []
    for key in cifar100_class:
        if key == "fine_label_names":
            # print(key)
            # print(len(cifar100_class[key]))
            for i, ele in enumerate(cifar100_class[key]):
                # print(ele)
                if ele == "boy":
                    # print(i)
                    list_delete.append(i)
                if ele == "girl":
                    # print(i)
                    list_delete.append(i)
                if ele == "baby":
                    # print(i)
                    list_delete.append(i)
                if ele == "man":
                    # print(i)
                    list_delete.append(i)
                if ele == "woman":
                    # print(i)
                    list_delete.append(i)
    # print(list_delete)

    def trans(img_PIL):

        def trans_(in_l):
            out_l = []
            out_l.append(in_l[2])
            out_l.append(in_l[1])
            out_l.append(in_l[0])
            return out_l

        w = img_PIL.size[0]
        h = img_PIL.size[1]
        out = []

        for i in range(w):
            out_line = []
            for j in range(h):
                np_pixel = []
                for ele in img_PIL.getpixel((i, j)):
                    np_pixel.append(int(ele))
                # print(np_pixel)
                # temp = np_pixel[0]
                # np_pixel[0] = np_pixel[2]
                # np_pixel[2] = temp
                np_pixel = trans_(np_pixel)
                out_line.append(np_pixel)
            out.append(out_line)

        # print(out)
        out = np.array(out)
        return out

    images = []
    for i, (image, label) in enumerate(train_dataset):
        # print(image.getpixel((1, 2)))
        if label not in list_delete:
            # print("there")
            image = trans(image)
            # print(len(image))
            images.append(image)

            name_N = i
            N = str(name_N) + ".jpg"
            N = os.path.join(os.path.join("dataset", "others"), N)
            cv.imwrite(N, image)

        if (i + 1) % 1000 == 0:
            print("Done %d" % (i + 1))
    # print(len(images))

    # cv.imwrite("test.jpg", images[36])

    def flip(img_name):
        img = PIL.Image.open(img_name)
        im_rotate = img.rotate(-90)
        im_rotate.save(img_name)

    path = os.path.join("dataset", "others")
    list_name = os.listdir(path)
    for i, n in enumerate(list_name):
        flip(os.path.join(path, n))
        if (i+1) % 1000 == 0:
            print("Done %d" % (i + 1))


if __name__ == "__main__":

    prepareFaceData()
    prepareOtherData()
    # flip("dataset/others/9.jpg")
