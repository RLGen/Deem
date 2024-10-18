import numpy as np
import joblib
from torch.utils.data import Dataset

def load_covertype_dataset(data_dir):
    samples_path = data_dir + "/covertype/samples_py3"
    targets_path = data_dir + "/covertype/targets_py3"

    # Load the data from the files
    X = joblib.load(samples_path)
    y = joblib.load(targets_path)
    y = y - 1

    return X, y

def classwise_split(dataset_targets, ratio, num_classes):
    # select {ratio*len(labels)} images from the images
    train_val_label = np.array(dataset_targets)
    train_indexes = []
    val_indexes = []

    for id in range(num_classes):
        indexes = np.where(train_val_label == id)[0]
        # print(len(indexes))
        np.random.shuffle(indexes)
        train_num = int(len(indexes)*ratio)
        train_indexes.extend(indexes[:train_num])
        val_indexes.extend(indexes[train_num:])
    
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes

class Convertype_train(Dataset):
    def __init__(self, data, labels, train_indexes, noise_ratio):
        # define the number of classes
        self.num_classes = 7
        
        # define the ratio
        self.noise_ratio = noise_ratio
        
        # define dATA
        self.data = data
        self.labels = labels

        # choose the data and label in dataset
        if train_indexes is not None:
            self.train_data = np.array(self.data)[train_indexes]
            self.train_labels = np.array(self.labels)[train_indexes]
            self.true_labels = np.array(self.labels)[train_indexes]
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
        dic = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:0}
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

class Convertype_val(Dataset):
    def __init__(self, data, labels, val_indexes):
        # define the number of classes
        self.num_classes = 7
        
        # define data
        self.val_data = np.array(data)[val_indexes]
        self.val_labels = np.array(labels)[val_indexes]

    def __getitem__(self, index):
        return self.val_data[index], self.val_labels[index]
    
    
    

# # # 指定你下载数据集的路径
# data_dir = "/data2/wyh-dataset/tabular-dataset"
# data, labels = load_covertype_dataset(data_dir)
        
# train_val_indexes, test_indexes = classwise_split(dataset_targets=labels, ratio=0.8, num_classes=7)
# train_val_data = np.array(data)[train_val_indexes]
# train_val_labels = np.array(labels)[train_val_indexes]

# train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=0.9, num_classes=7)
# train_dataset = Convertype_train(data=train_val_data, labels=train_val_labels, train_indexes=train_indexes, noise_ratio=0.2)
# val_dataset = Convertype_val(data=train_val_data, labels=train_val_labels,val_indexes=val_indexes)
# test_dataset = Convertype_val(data=data, labels=labels, val_indexes=test_indexes)
# print(len(train_dataset)) # 371843
# train_dataset.asymmetric_noise()
# print(train_dataset.noise_or_not.count(True)) # 74368对称 # 74366-7非对称
# # 调用自定义函数加载数据集
# X, y = load_covertype_dataset(data_dir)
# print(X.shape) # (581012, 54)
# print(y.shape) # (581012,)