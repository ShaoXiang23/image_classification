from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from PIL import Image

import csv
import os

'''
plt.figure()
for i in range(1,9):
    plt.subplot(2,4,i)
    plt.imshow(images[i-1])
    plt.xticks([])
    plt.yticks([])
plt.show()
'''

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

class MyDataset(Dataset):
    def __init__(self, csv_file_path):
        imgs = list()
        csv_file = csv.reader(open(csv_file_path, 'r'))
        for line in csv_file:
            # some data is ambigous
            if len(line[1]) <= 2:
                imgs.append(line)
        self.imgs = imgs

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        img = 'datasets/Train/'+ filename
        label = int(label) - 1

        return img, label

    def __len__(self):
        return len(self.imgs)

def img_rotate(img):
    img_90 = img.rotate(90)
    img_180 = img.rotate(180)
    img_270 = img.rotate(270)
    return [img, img_90, img_180, img_270]

def img_augmentation(img_name):
    resize_tool = Resize((512, 512))
    img_path = [img_name]
    img = Image.open(img_name).convert("RGB")
    img = resize_tool(img)
    img_ud = img.transpose(Image.FLIP_TOP_BOTTOM)
    results = img_path + img_rotate(img) + img_rotate(img_ud)
    return results

def img_enrich_x8(img_lists, csv_path="", new_img_path=""):
    image_path = img_lists[0].split('/')[2]
    label = str(img_lists[-1])
    csv_file = open(csv_path, 'a+', newline="")
    writer = csv.writer(csv_file)
    for i in range(8):
        full_path = new_img_path + str(i) + "_" + image_path
        file_path = str(i) + "_" + image_path
        img_lists[i+1].save(full_path)
        writer.writerow([file_path, label])

if __name__ == "__main__":
    # img = Image.open('C:/Users/Gsx/Desktop/1.jpg')
    result = img_augmentation('C:/Users/Gsx/Desktop/cat.jpg')[1:]
    plt.figure()
    for i in range(1, 9):
        plt.subplot(2, 4, i)
        plt.imshow(result[i - 1])
        plt.xticks([])
        plt.yticks([])
    plt.show()
# if __name__ == "__main__":
#     train_data = MyDataset('datasets/Train_label.csv')
#     train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
#     new_img_path = "C:/Users/Gsx/Desktop/images/"
#     csv_path = 'C:/Users/Gsx/Desktop/1021.csv'
#
#     for i, (images, labels) in enumerate(train_loader):
#         image = images[0]
#         labels = labels.numpy().tolist()[0]
#         # print(len(img_augmentation(img_name=image)))
#         img_enrich_x8(img_augmentation(img_name=image) + [labels], csv_path=csv_path, new_img_path=new_img_path)
#

