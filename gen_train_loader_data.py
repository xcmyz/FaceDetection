import torch
import torchvision.transforms.functional as TF
# import json
import os
import random
import numpy as np
from PIL import Image
from hyperparameters import length_sample


def gen_train_loader(batch_size=50):
    cnt = 0
    cnt_ = 0
    cnt_1 = 0
    cnt_2 = 0

    dataset_P = {}
    dataset_N = {}

    pF = os.path.join("data", "dataset")
    pDP = os.path.join("database", "P")
    pDN = os.path.join("database", "N")
    path_P = os.path.join(pF, "face")
    path_N = os.path.join(pF, "others")

    # Process face
    list_all_PData = []

    list_face_name = os.listdir(path_P)
    list_index_face = [i for i in range(len(list_face_name))]
    random.shuffle(list_index_face)
    for index in list_index_face:
        im = Image.open(os.path.join(path_P, list_face_name[index]))
        # print(list_face_name[index])
        im_1 = TF.hflip(im)
        tensor1 = TF.to_tensor(im)
        tensor2 = TF.to_tensor(im_1)
        l_1 = tensor1.tolist()
        l_2 = tensor2.tolist()
        cnt_ = cnt_ + 2
        # print(len(l_1))
        # print(len(l_1[0]))
        # print(len(l_1[0][0]))
        list_all_PData.append(l_1)
        list_all_PData.append(l_2)

        cnt = cnt + 1
        if cnt % (length_sample/2) == 0:
            print("Done %d" % cnt)

            temp_num = cnt_1
            N = str(cnt_1) + ".txt"
            name = os.path.join(pDP, N)
            dataset_P.update({"P": list_all_PData})
            with open(name, 'w') as file:
                file.write(str(dataset_P))

            dataset_P.clear()
            list_all_PData.clear()
            cnt_1 = cnt_1 + 1

    # print(len(list_all_PData))
    # dataset.update({"P": list_all_PData})
    # tensors_P = torch.tensor(list_all_PData)

    # Process face
    cnt = 0
    list_all_NData = []
    # print(cnt_)

    list_others_name = os.listdir(path_N)
    list_index_others = [i for i in range(len(list_others_name))]
    random.shuffle(list_index_others)
    for num, index in enumerate(list_index_others):
        if num < cnt_:
            im = Image.open(os.path.join(path_N, list_others_name[index]))
            # print(os.path.join(path_N, list_others_name[index]))
            # im_1 = TF.hflip(im)
            tensor = TF.to_tensor(im)
            # tensor2 = TF.to_tensor(im_1)
            l = tensor.tolist()
            # l_2 = tensor2.tolist()
            # print(len(l_1))
            # print(len(l_1[0]))
            # print(len(l_1[0][0]))
            list_all_NData.append(l)
            # list_all_PData.append(l_2)

            cnt = cnt + 1
            if cnt % length_sample == 0:
                print("Done %d" % cnt)

                temp_num = cnt_2
                N = str(cnt_2) + ".txt"
                name = os.path.join(pDN, N)
                dataset_N.update({"N": list_all_NData})
                with open(name, 'w') as file:
                    file.write(str(dataset_N))

                dataset_N.clear()
                list_all_NData.clear()
                cnt_2 = cnt_2 + 1

    # print(len(list_all_NData))
    # dataset.update({"N": list_all_NData})
    # js_dataset = json.dumps(dataset)

    # for i in range(10000):
    #     for i in range(10000):
    #         a = 1
    #     print("###############")

    # with open('dataset.txt', 'w') as file:
    #     file.write(str(dataset))
    # file.close()
    # tensors_N = torch.tensor(list_all_NData)


if __name__ == "__main__":
    gen_train_loader()
