from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import csv
import os

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        # padding
        ratio = self.size[0] / self.size[1]
        w, h = img.size
        if w / h < ratio:
            t = int(h * ratio)
            w_padding = (t - w) // 2
            img = img.crop((-w_padding, 0, w+w_padding, h))
        else:
            t = int(w / ratio)
            h_padding = (t - h) // 2
            img = img.crop((0, -h_padding, w, h+h_padding))

        img = img.resize(self.size, self.interpolation)

        return img

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_loader_keep(path):
    resize_tool = Resize((512, 512))
    img = Image.open(path).convert('RGB')
    img = resize_tool(img)
    
    return img

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    transforms.ToTensor(),
    # transforms.RandomCrop(96),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class MyDataset(Dataset):
    def __init__(self, csv_file_path, transform=transforms, loader=default_loader):
        imgs = list()
        csv_file = csv.reader(open(csv_file_path, 'r'))
        for line in csv_file:
            if len(line[1]) <= 2:
                imgs.append(line)
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = self.loader('datasets/Train/'+ filename)
        if self.transform is not None:
            img = self.transform(img)
        label = int(label) - 1

        return img, label

    def __len__(self):
        return len(self.imgs)

# if __name__ == "__main__":
#     train_data = MyDataset('datasets/Train_label.csv')
#     train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=1,
#                           pin_memory=True)
#
#     labels = []
#     images = []
#     for i, (image_path, label) in enumerate(train_loader, 0):
#         image_path = image_path[0]
#         images.append(image_path)
#         label = label.numpy().tolist()[0]
#         # print(image_path, label)
#         labels.append(label)
#
#     x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=0)
#     for ele in zip(x_train, y_train):
#         print(list(ele))
    # class_count = {}
    # for i in set(labels):
    #     class_count[i] = labels.count(i)
    # res = sorted(class_count.items(), key=lambda class_count: class_count[1], reverse=True)
    #
    # _class = []
    # _num = []
    # for ele in res:
    #     _class.append(ele[0])
    #     _num.append(ele[1])
    #
    # print(_num)
