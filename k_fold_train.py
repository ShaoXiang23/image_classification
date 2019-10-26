from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from cloud_dataloader import MyDataset
import matplotlib.pyplot as plt
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torchvision
import numpy as np
import copy
import os
from models.resnet import resnet50


classes = ["中云-高积云-絮状高积云", "中云-高积云-透光高积云",
           "中云-高积云-荚状高积云", "中云-高积云-积云性高积云",
           "中云-高积云-蔽光高积云", "中云-高积云-堡状高积云",
           "中云-高层云-透光高层云", "中云-高层云-蔽光高层云",
           "高云-卷云-伪卷云", "高云-卷云-密卷云",
           "高云-卷云-毛卷云", "高云-卷云-钩卷云",
           "高云-卷积云-卷积云", "高云-卷层云-匀卷层云",
           "高云-卷层云-毛卷层云", "低云-雨层云-雨层云",
           "低云-雨层云-碎雨云", "低云-积云-碎积云",
           "低云-积云-浓积云", "低云-积云-淡积云",
           "低云-积雨云-鬃积雨云", "低云-积雨云-秃积雨云",
           "低云-层云-碎层云", "低云-层云-层云",
           "低云-层积云-透光层积云", "低云-层积云-荚状层积云",
           "低云-层积云-积云性层积云", "低云-层积云-蔽光层积云",
           "低云-层积云-堡状层积云"]

k1_path = ('datasets/Train_kf_1.csv', 'datasets/Test_kf_1.csv')
k2_path = ('datasets/Train_kf_2.csv', 'datasets/Test_kf_2.csv')
k3_path = ('datasets/Train_kf_3.csv', 'datasets/Test_kf_3.csv')
k4_path = ('datasets/Train_kf_4.csv', 'datasets/Test_kf_4.csv')
k5_path = ('datasets/Train_kf_5.csv', 'datasets/Test_kf_5.csv')
kfold_lists = [k1_path, k2_path, k3_path, k4_path, k5_path]

# function parameters
model_path = 'weights/resnet50'
train_batch_size = 1
verify_batch_size = 1


# parameters
k = 5
epochs = 3
lr = 1e-4
weight_decay = 0
momentum = 0.9
global_step = 0

if __name__ == "__main__":

    # record fold i the min_loss
    loss_info = []

    # K-time training
    for fold in range(k):
        model = resnet50(pretrained=True, my_classes=29)
        if torch.cuda.is_available():
            model = model.cuda()

        # set corresponding train_data and test_data
        train_data = MyDataset(kfold_lists[fold][0])
        train_loader = DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)

        # min_loss
        min_loss = [0.9, 0.9]

        # training
        for epoch in range(epochs):
            print('---------- Epoch %d: training ---------' % epoch)
            model.train()

            # adjust learning rate by epoch
            if epoch == 0:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif epoch == 1:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr/50, weight_decay=weight_decay)
            else:
                optimizer = torch.optim.Adam(model.parameters(), lr=lr/50, weight_decay=weight_decay)

            for step, (data, target) in enumerate(train_loader):
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()

                optimizer.zero_grad()
                output = model(data)

                # special loss: binary_cross_entroy
                loss = torch.nn.BCELoss(output, target)
                loss.backward()
                optimizer.step()

                global_step += 1

                # step verify
                if step % 10 == 0:
                    model.eval()
                    if torch.cuda.is_available():
                        model = model.cuda()

                    # corresponding verify datasets for kfold
                    verify_data = MyDataset(kfold_lists[fold][1])
                    verify_loader = DataLoader(dataset=verify_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
                    total_vloss = 0
                    for vstep, (vdata, vtarget) in enumerate(verify_loader):
                        if torch.cuda.is_available():
                            vdata = vdata.cuda()
                            vtarget = vtarget.cuda()

                        voutput = model(vdata)
                        vloss = torch.nn.BCELoss(voutput, vtarget)
                        total_vloss += vloss.data[0]

                    vloss = total_vloss / (10211 * 0.2 / verify_batch_size)

                    model.train()
                    if torch.cuda.is_available():
                        model = model.cuda()

                    print('{} Fold{} Epoch{} Step{}: [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}\tValidate Loss: {:.6f}'.format(
                            model_path.split('/')[-1],
                            fold,
                            epoch,
                            global_step,
                            step * train_batch_size,
                            len(train_loader.dataset),
                            100. * step / len(train_loader),
                            loss.data[0],
                            vloss))

                    if vloss < min_loss[1]:
                        min_loss[1] = vloss
                        min_loss[0] = loss.data[0]


        # save model
        model_save = copy.deepcopy(model)
        torch.save(model_save.cpu(), os.path.join(model_path, 'resnet50_fold_%d.model' % (fold)))
        loss_info.append(min_loss)

    print('-' * 20)
    print(model_path.split('/')[-1] + ":")
    for i, l in enumerate(loss_info):
        print("Fold %d: Train loss:%f\tVerify loss:%f" % (i, l[0], l[1]))
