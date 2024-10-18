# Define food101 dataset for training
# Author: wyh
# Date: 2024.1.26

import numpy as np
from PIL import Image
from torchvision.datasets import Food101

# don't use args to pass parameter
class FOOD101_Train(Food101):
    # init the data and labels including hard and soft labels
    def __init__(self, train_root, train_indexes=None, noise_ratio=0, split="train", transform=None, target_transform=None, download=False):
        super(FOOD101_Train, self).__init__(train_root, split=split, transform=transform, target_transform=target_transform, download=download)
        # define the number of classes
        self.num_classes = 101
        # define the ratio
        self.noise_ratio = noise_ratio

        # choose the data and label in dataset
        if train_indexes is not None:
            self.train_data = np.array(self._image_files)[train_indexes]
            self.train_labels = np.array(self._labels)[train_indexes]
            self.true_labels = np.array(self._labels)[train_indexes]
        self.soft_labels = np.zeros((len(self.train_labels), self.num_classes), dtype=np.float32)
        self.noise_or_not = [False] * len(self.train_data)  # 初始化为全部为False

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
        # 暂时的dic，键值对可更改
        # dic = {9: 1, 2: 0, 3: 5, 5: 3, 4: 7}
        dic = {i: (i+1) for i in range(100)}
        dic[100] = 0
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
        
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label, true_label, is_noise, soft_label, index
    
    def __len__(self) -> int:
        return len(self.train_labels)

class FOOD101_Val(Food101):
    def __init__(self, val_root, val_indexes, split="train", transform=None, target_transform=None, download=False):
        super(FOOD101_Val, self).__init__(val_root, split=split, transform=transform, target_transform=target_transform, download=download)
        # self.train_labels & self.train_data are the attrs from tv.datasets.SVHN
        self.val_data = np.array(self._image_files)[val_indexes]
        self.val_labels = np.array(self._labels)[val_indexes]

    def __getitem__(self, index):
        img, label = self.val_data[index], self.val_labels[index]
        img = Image.open(img).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return img, label
    
    def __len__(self) -> int:
        return len(self.val_labels)
    
# from torchvision.datasets import Food101
# import torchvision.transforms as transforms
# from torchvision.transforms import ToTensor
# data_root = "/data2/wyh-dataset/image-dataset" + "/Food101"
# # load dataset
# dataset = Food101(root=data_root, split="train", download=False, transform=ToTensor())
# print(dataset)

# input_size = 224

# transform = transforms.Compose([
#     transforms.RandomResizedCrop(input_size),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# import torch
# dataset = Food101(root=data_root, split="train", download=False, transform=transform)
# test_dataset = Food101(root=data_root, split="test", download=False, transform=ToTensor())

# for i in range(len(dataset)):
#     print("样本大小:", dataset[i][0].size())
