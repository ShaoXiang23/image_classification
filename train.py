from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from cloud_dataloader import MyDataset
from se_resnet import se_resnet50
from resnet import resnext101_32x8d
from ResNet_20191020 import resnet50
from ResNet_20191020 import resnet152
from models.resnext_wsl import resnext101_32x16d_wsl
import matplotlib.pyplot as plt
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torchvision
import numpy as np
from torch.optim import lr_scheduler
from models.resnet50_se import resnet50_se
from models.resneXt import resnext50_32x4d

def adjust_learning_rate(epoch):
    lr = 0.0001

    if epoch > 50:
        lr = lr / 10
    elif epoch > 100:
        lr = lr / 50

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

train_data = MyDataset('datasets/Train_enrich_train.csv')
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

test_data = MyDataset('datasets/Train_enrich_test.csv')
test_loader = DataLoader(dataset=test_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)

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

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def save_models(epoch, model):
    torch.save(model.state_dict(), "weights/wsl_model_{}.model".format(epoch))
    print("Checkpoint saved.")

def evalution(testloader=None, net=None):
    # Evaluation
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (labels == predicted).sum().item()

    print("Accuracy of the network on the 10000 test images: %d %%" % (100 * correct / total))
    return correct / total


def evalution_by_class(classes, testloader=None, net=None):
    # Evaluation by class
    n_classes = len(classes)
    class_correct, class_total = [0] * n_classes, [0] * n_classes

    with torch.no_grad():
        for images, labels in testloader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            is_correct = (labels == predicted).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                class_correct[label] += is_correct[i].item()

    for i in range(n_classes):
        print("Accuracy of %5s: %.2f %%" % (classes[i], 100.0 * class_correct[i] / class_total[i]))

# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# net = se_resnet50(num_classes=len(classes))
# net = resnext101_32x8d(num_classes=len(classes))
# net = resnet152(pretrained=True)
# net = resnet50_se(pretrained=True, my_classes=29)
# net = resnext50_32x4d(pretrained=True, my_classes=29)
net = resnext101_32x16d_wsl()
net.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 29))
pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/ig_resnext101_32x16-c6f796b0.pth')
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
net.load_state_dict(model_dict)
# print(net)
# exit()
if torch.cuda.is_available():
    net = net.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4, weight_decay=0.0001)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

num_epoch = 5
for epoch in range(num_epoch):
    best_acc = 0.0
    _loss = 0.0
    if epoch > 1:
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4 / 50, weight_decay=0.0001)
    elif epoch > 3:
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-4 / 1000, weight_decay=0.0001)
    for i, (inputs, labels) in enumerate(train_loader, 0):
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _loss += loss.item()
        if i % 100 == 99: #每200步打印一次损失值
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, _loss / 100))
            _loss = 0.0
    # adjust_learning_rate(epoch=epoch)
    test_acc = evalution(testloader=test_loader, net=net)
    # if epoch % 5 == 0:
    if test_acc > best_acc:
        save_models(epoch=epoch, model=net)
        best_acc = test_acc
    # evalution_by_class(classes=classes, testloader=test_loader, net=net)
print("Finished Training.")
