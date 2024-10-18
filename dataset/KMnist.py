# Define KMNIST dataset for training
# Author: wyh
# Date: 2024.1.26

import numpy as np
from PIL import Image
from torchvision.datasets import KMNIST
from torch.utils.data import Dataset

# define Kmnist dataset for training and give functions to make noise labels
classes = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

# don't use args to pass parameter
class KMNIST_Train(Dataset):
    # init the data and labels including hard and soft labels
    def __init__(self, noise_ratio, train_root, train_indexes=None, train=True, transform=None, target_transform=None, download=False):
        dataset = KMNIST(root=train_root, train=train, download=download, transform=transform, target_transform=target_transform)
        # define the number of classes
        self.num_classes = 10
        # define the ratio
        self.noise_ratio = noise_ratio
        self.transform=transform
        self.target_transform=target_transform
        
        # choose the data and label in dataset
        if train_indexes is not None:
            self.train_data = np.array(dataset.data)[train_indexes]
            self.true_labels = np.array(dataset.targets)[train_indexes]
            self.train_labels = np.array(dataset.targets)[train_indexes]
        else:
            self.train_data = np.array(dataset.data)
            self.true_labels = np.array(dataset.targets)
            self.train_labels = np.array(dataset.targets)
        self.soft_labels = np.zeros((len(self.train_labels), self.num_classes), dtype=np.float32)
        self.noise_or_not = [False] * len(self.train_data)  # 初始化为全部为False

    def __len__(self):
        return len(self.train_labels)
    
    # make symmetric noise labels
    # The difference between generating random noise overall and generating noise for each class is not significant
    def symmetric_noise(self):
        # generate random noise across the dataset
        idxes = np.random.permutation(len(self.true_labels))
        noise_num = int(len(self.true_labels) * self.noise_ratio)
        for i in range(len(idxes)):
            if i < noise_num:
                # a randomly generated label that differs from the privious label
                exclude_class = self.true_labels[idxes[i]]
                label_sym = np.random.choice(np.delete(np.arange(self.num_classes), exclude_class), 1)[0]
                self.train_labels[idxes[i]] = label_sym
                self.noise_or_not[idxes[i]] = True
            self.soft_labels[idxes[i]][self.train_labels[idxes[i]]] = 1
    
    # make asymmetric noise labels
    def asymmetric_noise(self):
        # dic = {4:5, 0:4, 5:6, 6:9}
        dic = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:0}
        for i in range(self.num_classes):
            idxes = np.where(self.true_labels == i)[0]
            np.random.shuffle(idxes)
            noise_num = int(int(len(idxes)) * self.noise_ratio) #noise_ratio
            for j in range(len(idxes)):
                if j< noise_num and i in dic:
                    self.train_labels[idxes[j]] = dic[i]
                    self.noise_or_not[idxes[j]] = True
                self.soft_labels[idxes[j]][self.train_labels[idxes[j]]] = 1

    def __getitem__(self, index):
        img, label, true_label, soft_label = self.train_data[index], self.train_labels[index], self.true_labels[index], self.soft_labels[index]
        
        # 判断是否是噪音数据
        is_noise = False
        if label != true_label:
            is_noise = True
        
        
        img = Image.fromarray(img.numpy(), mode="L")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, true_label, is_noise, soft_label, index

class KMNIST_Val(Dataset):
    def __init__(self, val_root, val_indexes, train=True, transform=None, target_transform=None, download=False):
        # self.train_labels & self.train_data are the attrs from tv.datasets.CIFAR10
        dataset = KMNIST(root=val_root, train=train, download=download, transform=transform, target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.val_data = np.array(dataset.data)[val_indexes]
        self.val_labels = np.array(dataset.targets)[val_indexes]
    
    def __len__(self):
        return len(self.val_labels)

    def __getitem__(self, index):  
        img, label = self.val_data[index], self.val_labels[index]
        # doing this so that it is consistent with all other datasets.
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
