import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import os
import time
import numpy as np
from network import ResNet, ResidualBlock
from hyperparameters import *
import matplotlib.pyplot as plt
from PIL import Image


def train_loader(file_name_P, file_name_N, batch_size=50):
    train_loader = []
    # print(file_name_P)
    # print(file_name_N)
    with open(file_name_P, 'r') as f_P:
        str_P = f_P.read()
        dict_P = eval(str_P)
    dataset_P = dict_P["P"]
    # print(len(dataset_P))
    # print(dataset_P)
    # print(len(dataset_P[0]))

    with open(file_name_N, 'r') as f_N:
        str_N = f_N.read()
        dict_N = eval(str_N)
    dataset_N = dict_N["N"]
    # print(len(dataset_N))

    one_batch_im = []
    one_batch_label = []
    if len(dataset_P) == len(dataset_N):
        for i in range(len(dataset_P)):
            # print(1)
            # one_batch.append((dataset_P[i], 0))
            # one_batch.append((dataset_N[i], 1))
            one_batch_im.append(dataset_P[i])
            one_batch_im.append(dataset_N[i])
            one_batch_label.append(0)
            one_batch_label.append(1)
            # print(one_batch_label)
            if len(one_batch_im) == batch_size:
                # print(one_batch_im)
                # print(one_batch_label)
                # one_batch_im = torch.Tensor(one_batch_im)
                # one_batch_label = torch.Tensor(one_batch_label)
                temp_im = one_batch_im.copy()
                temp_label = one_batch_label.copy()
                temp_im = torch.Tensor(temp_im)
                # temp_label = torch.Tensor(temp_label, dtype=torch.int32)
                temp_label = torch.Tensor(temp_label)
                temp_label = temp_label.long()
                # print(temp_label)
                train_loader.append((temp_im, temp_label))
                one_batch_im.clear()
                one_batch_label.clear()
    else:
        print("ERROR")

    # train_loader = torch.Tensor(train_loader)
    return train_loader


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == "__main__":
    # path_data_P = os.path.join("database", "P")
    # path_data_N = os.path.join("database", "N")

    # # Test
    # train_loader = train_loader(os.path.join(path_data_P, "0.txt"),
    #                             os.path.join(path_data_N, "0.txt"), batch_size=batchsize)
    # print(len(train_loader))
    # print(len(train_loader[0]))
    # print(len(train_loader[0][0]))
    # print(train_loader[0][1])
    # print(len(train_loader[0][0][0]))
    # print(len(train_loader[0][0][0][0]))

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = ResNet(ResidualBlock, layers, number_classes).to(device)
    print("Model has been defined.")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    # # Test
    # path_data_P = os.path.join("database", "P")
    # path_data_N = os.path.join("database", "N")
    # train_loader = train_loader(os.path.join(path_data_P, "0.txt"),
    #                             os.path.join(path_data_N, "0.txt"), batch_size=batchsize)
    # total_step = len(train_loader)
    # curr_lr = learning_rate
    # for epoch in range(num_epochs):
    #     for i, (images, labels) in enumerate(train_loader):
    #         # # Test
    #         # print(len(images))
    #         # print(len(images[0]))
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         # print(labels)

    #         # Forward pass
    #         outputs = model(images)
    #         loss = criterion(outputs, labels)

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if (i+1) % 1 == 0:
    #             print("Epoch [{}/{}], Step [{}/{}] Loss: {:.6f}"
    #                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    #     # Decay learning rate
    #     if (epoch+1) % 20 == 0:
    #         curr_lr /= 3
    #         update_lr(optimizer, curr_lr)

    # im = Image.open("test.jpg")
    # tensor_test = TF.to_tensor(im)
    # images = tensor_test.to(device)
    # model.eval()
    # print(model(images))

    path_data_P = os.path.join("database", "P")
    path_data_N = os.path.join("database", "N")

    loss_list = []
    ts = time.clock()
    print("Train begin...")
    for epoch in range(num_epochs):
        count = 0
        # Get train loader
        P_file_list = os.listdir(path_data_P)
        N_file_list = os.listdir(path_data_N)
        for cnt in range(len(P_file_list)):
            # tfs = time.clock()
            P_file = os.path.join(path_data_P, P_file_list[cnt])
            N_file = os.path.join(path_data_N, N_file_list[cnt])
            # print(P_file)
            # print(N_file)
            trainloader = train_loader(P_file, N_file, batch_size=batchsize)
            # tfe = time.clock()
            # print("File loaded use: {:.2f}".format((tfe-tfs)))
            total_step = len(trainloader) * len(os.listdir(path_data_P))

            # tcs = time.clock()
            for i, (images, labels) in enumerate(trainloader):
                # # Test
                # print(len(images))
                # print(len(images[0]))
                images = images.to(device)
                labels = labels.to(device)
                # print(labels)
                # print(np.shape(images))

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                count = count + 1

                if count % 10 == 0:
                    loss_list.append(loss.item())
                    # print(loss_list)
                if count % 30 == 0:
                    tt = time.clock()
                    print("Epoch [{}/{}], Step [{}/{}] Loss: {:.6f} Time: {:.3f}"
                          .format(epoch+1, num_epochs, count, total_step, loss.item(), (tt - ts)))
            # tce = time.clock()
            # print("Calculate use: {:.2f}".format((tce-tcs)))

        # Decay learning rate
        if (epoch + 1) % 20 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)

    print("Train end.")

    # Save the model checkpoint
    path_model = os.path.join("model", "resnet.ckpt")
    torch.save(model.state_dict(), path_model)
    print("Model saved.")

    plt.figure()
    plt.plot([i for i in range(len(loss_list))], loss_list)
    plt.savefig("loss.jpg")
