from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader, Dataset
import csv

class MyDataset(Dataset):
    def __init__(self, csv_file_path):
        imgs = list()
        csv_file = csv.reader(open(csv_file_path, 'r'))
        for line in csv_file:
            if len(line[1]) <= 2:
                imgs.append(line)
        self.imgs = imgs

    def __getitem__(self, index):
        filename, label = self.imgs[index]
        label = int(label) - 1

        return filename, label

    def __len__(self):
        return len(self.imgs)

def make_k_fold_csv(csv_path_train, csv_path_test, index_lists, images, labels):
    # make train_csv
    csv_file_train = open(csv_path_train, 'a+', newline="")
    train_writer = csv.writer(csv_file_train)
    train_index = index_lists[0].tolist()
    for index in train_index:
        train_writer.writerow([images[index], labels[index]])
    # # make test_csv
    csv_file_test = open(csv_path_test, 'a+', newline="")
    test_writer = csv.writer(csv_file_test)
    test_index = index_lists[1].tolist()
    for index in test_index:
        test_writer.writerow([images[index], labels[index]])


if __name__ == "__main__":
    train_data = MyDataset('datasets/Train_label.csv')
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=1,
                          pin_memory=True)

    labels = []
    images = []
    for i, (image_path, label) in enumerate(train_loader, 0):
        image_path = image_path[0]
        images.append(image_path)
        label = label.numpy().tolist()[0]
        # print(image_path, label)
        labels.append(label)
    # print(len(images), len(labels))

    kf = KFold(n_splits=5, shuffle=True, random_state=7)
    kf.get_n_splits(images)
    print(kf)
    index_lists = []
    for train_index, test_index in kf.split(images):
        index_lists.append([train_index, test_index])
    print(index_lists)

    for i in range(5):
        train_csv_path = 'datasets/Train_kf_' + str(i+1) + '.csv'
        test_csv_path = 'datasets/Test_kf_' + str(i+1) + '.csv'
        print(train_csv_path, test_csv_path)
        make_k_fold_csv(train_csv_path, test_csv_path, index_lists[i], images, labels)

    '''Verify'''
    # import csv
    #
    # sum = []
    # csv_file = csv.reader(open("datasets/Test_kf_1.csv", 'r'))
    # for line in csv_file:
    #     sum.append(line[0])
    # csv_file = csv.reader(open("datasets/Test_kf_2.csv", 'r'))
    # for line in csv_file:
    #     sum.append(line[0])
    # csv_file = csv.reader(open("datasets/Test_kf_3.csv", 'r'))
    # for line in csv_file:
    #     sum.append(line[0])
    # csv_file = csv.reader(open("datasets/Test_kf_4.csv", 'r'))
    # for line in csv_file:
    #     sum.append(line[0])
    # csv_file = csv.reader(open("datasets/Test_kf_5.csv", 'r'))
    # for line in csv_file:
    #     sum.append(line[0])
    #
    # sum = sorted(sum)
    # images = sorted(images)
    # print(sum)
    # print(images)
