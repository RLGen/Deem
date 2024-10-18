import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class HepmassDataset():
    def __init__(self, data_dir, train_ratio):
        train_val = pd.read_csv(data_dir+"/HEPMASS/all_train.csv.gz")
        test = pd.read_csv(data_dir+"/HEPMASS/all_test.csv.gz")
        scaler = StandardScaler()
        # 对 mass 列进行标准化处理可以使得不同粒子的质量具有相似的尺度
        train_val['mass'] = scaler.fit_transform(train_val['mass'].values.reshape(-1, 1))
        test['mass'] = scaler.fit_transform(test['mass'].values.reshape(-1, 1))
        self.train_val_data = train_val.drop(['# label'], axis=1).values
        self.train_val_labels = train_val['# label'].values

        self.test_data = test.drop(['# label'], axis=1).values
        self.test_labels = test['# label'].values

        self.train_data, self.val_data, self.train_labels, self.val_labels = train_test_split(self.train_val_data, self.train_val_labels, test_size=train_ratio, random_state=42)
    
    def get_train_dataset(self):
        return self.train_data, self.train_labels
    
    def get_val_dataset(self):
        return self.val_data, self.val_labels
    
    def get_test_dataset(self):
        return self.test_data, self.test_labels

class Hepmass_train(Dataset):
    def __init__(self, train_data, train_labels, noise_ratio):
        # define the number of classes
        self.num_classes = 2

        # define data
        self.train_data = train_data
        self.true_labels = train_labels.astype(int)
        self.train_labels = train_labels.astype(int)
        self.soft_labels = np.zeros((len(self.train_labels), self.num_classes), dtype=np.float32)


        # define noise ratio
        self.noise_ratio = noise_ratio
        self.noise_or_not = [False] * len(self.train_labels)  # 初始化为全部为False
    
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
        dic = {0:1, 1:0}
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
        # 判断是否是噪音数据
        label = self.train_labels[index]
        true_label = self.true_labels[index]
        is_noise = False
        if label != true_label:
            is_noise = True
        return self.train_data[index], self.train_labels[index], self.true_labels[index], is_noise, self.soft_labels[index], index
    
    def __len__(self) -> int:
        return len(self.train_labels)

class Hepmass_val(Dataset):
    def __init__(self, val_data, val_labels):
        # define the number of classes
        self.num_classes = 2

        # define data
        self.val_data = val_data
        self.val_labels = val_labels.astype(int)
    
    def __getitem__(self, index):
        return self.val_data[index], self.val_labels[index]
    
    def __len__(self) -> int:
        return len(self.val_labels)


# train = pd.read_csv("/data2/wyh-dataset/tabular-dataset/HEPMASS/all_train.csv.gz")
# # train.info()
# scaler = StandardScaler()
# # 对 mass 列进行标准化处理可以使得不同粒子的质量具有相似的尺度
# train['mass'] = scaler.fit_transform(train['mass'].values.reshape(-1, 1))
# X = train.drop(['# label'], axis=1)
# y = train['# label']
# X=list(X)
# y = list(y)
# X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)
# print(X)
# print(y)
# n_features = X.values.shape[1]
# print(n_features)
# dataset = HepmassDataset(data_dir="/data2/wyh-dataset/tabular-dataset")
# train_data, train_labels = dataset.get_train_dataset()
# val_data, val_labels = dataset.get_val_dataset()
# test_data, test_labels = dataset.get_test_dataset()

# train_dataset = Hepmass_train(train_data=train_data, train_labels=train_labels, noise_ratio=0.2)
# val_dataset = Hepmass_val(val_data=val_data, val_labels=val_labels)
# test_dataset = Hepmass_val(val_data=test_data, val_labels=val_labels)
# print(len(train_dataset))
# train_dataset.asymmetric_noise()
# print(train_dataset.noise_or_not.count(True)) 


    
