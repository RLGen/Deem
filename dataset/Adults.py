import pandas as pd
import numpy as np
from torch.utils.data import Dataset

def load_adult_dataset(data_dir):
    samples_path = data_dir + "/adult/train.csv"
    targets_path = data_dir + "/adult/test.csv"

    # Load the data from the files
    df_train = pd.read_csv(samples_path, header=None, skiprows = 0, names=["age","workclass","fnlwgt","education","education_num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","money"])
    df_test = pd.read_csv(targets_path, header=None, skiprows = 0, names=["age","workclass","fnlwgt","education","education_num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","money1"])
    # df_train.info()
    return df_train, df_test

def dataCleaning(dataSet):
    # Delete unrelated rows based on conditions
    # dataSet.drop('fnlwgt',axis=1, inplace=True)         #fnlgwt
    dataSet.drop('education',axis=1, inplace=True)      #Education
    # dataSet.drop('capital-gain',axis=1, inplace=True)   #Capital Gain
    # dataSet.drop('capital-loss',axis=1, inplace=True)   #Capital Loss
    dataSet = dataSet.replace(' ?', np.nan)
    
    # Missing values handling, using mode imputation (mode() method to calculate the mode
    dataSet.fillna(value={'workclass':dataSet['workclass'].mode()[0],   #Workclass
                          'occupation':dataSet['occupation'].mode()[0],   #Occupation  
                          'native-country':dataSet['native-country'].mode()[0]}, #Native country
              inplace = True)  

    # Discretization of categorical data
    workclass = {' State-gov': 0,' Self-emp-not-inc': 1,' Private': 2,' Federal-gov': 3,' Local-gov': 4,' Self-emp-inc': 5, ' Without-pay': 6, ' Never-worked': 7}
    maritalStatus = {' Never-married': 0,' Married-civ-spouse': 1,' Divorced': 2,' Married-spouse-absent': 3, ' Separated': 4, ' Married-AF-spouse': 5, ' Widowed': 6}
    occupation = {' Adm-clerical': 0, ' Exec-managerial': 1, ' Handlers-cleaners': 2, ' Prof-specialty': 3, ' Other-service': 4, ' Sales': 5, ' Craft-repair': 6, ' Transport-moving': 7, ' Farming-fishing': 8, ' Machine-op-inspct': 9, ' Tech-support': 10, ' Protective-serv': 11,' Armed-Forces': 12, ' Priv-house-serv': 13}
    relationship = {' Not-in-family': 0, ' Husband': 1, ' Wife': 2, ' Own-child': 3, ' Unmarried': 4, ' Other-relative': 5}
    race = {' White': 0, ' Black': 1, ' Asian-Pac-Islander': 2, ' Amer-Indian-Eskimo': 3, ' Other': 4}
    sex = {' Male': 0, ' Female': 1}
    nativeCountry = {' United-States': 0, ' Cuba': 1, ' Jamaica': 2, ' India': 3, ' Mexico': 4, ' South': 5, ' Puerto-Rico': 6, ' Honduras': 7, ' England': 8, ' Canada': 9, ' Germany': 10, ' Iran': 11, ' Philippines': 12, ' Italy': 13, ' Poland': 14, ' Columbia': 15, ' Cambodia': 16, ' Thailand': 17, ' Ecuador': 18, ' Laos': 19, ' Taiwan': 20, ' Haiti': 21, ' Portugal': 22, ' Dominican-Republic': 23, ' El-Salvador': 24, ' France': 25, ' Guatemala': 26, ' China': 27, ' Japan': 28, ' Yugoslavia': 29, ' Peru': 30, ' Outlying-US(Guam-USVI-etc)': 31, ' Scotland': 32, ' Trinadad&Tobago': 33, ' Greece': 34, ' Nicaragua': 35, ' Vietnam': 36, ' Hong': 37, ' Ireland': 38, ' Hungary': 39, ' Holand-Netherlands': 40}
    money = {' <=50K': 0, ' >50K': 1}
    money1 = {' <=50K.': 0, ' >50K.': 1}

    dataSet['workclass'] = dataSet['workclass'].map(workclass)
    dataSet['marital-status'] = dataSet['marital-status'].map(maritalStatus)
    dataSet['occupation'] = dataSet['occupation'].map(occupation)
    dataSet['relationship'] = dataSet['relationship'].map(relationship)
    dataSet['race'] = dataSet['race'].map(race)
    dataSet['sex'] = dataSet['sex'].map(sex)
    dataSet['native-country'] = dataSet['native-country'].map(nativeCountry)
    if 'money' in dataSet.columns:
        dataSet['money'] = dataSet['money'].map(money)
    if 'money1' in dataSet.columns:
        dataSet['money1'] = dataSet['money1'].map(money1)

    if isinstance(dataSet, pd.DataFrame):
        return dataSet

class Adults():
    def __init__(self, data_dir):
        df_train, df_test = load_adult_dataset(data_dir=data_dir)
        self.train_data = dataCleaning(df_train)
        self.test_data = dataCleaning(df_test) 

    def get_train_dataset(self):
        return self.train_data
    
    def get_test_dataset(self):
        return self.test_data
    
class Adults_train(Dataset):
    def __init__(self, train_data, noise_ratio):
        # define the number of classes
        self.num_classes = 2

        # define data
        self.train_data = train_data[:, :-1]
        self.true_labels = np.array(train_data[:,-1])
        self.train_labels = np.array(train_data[:,-1])
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

class Adults_val(Dataset):
    def __init__(self, val_data):
        # define the number of classes
        self.num_classes = 2

        # define data
        self.val_data = val_data[:, :-1]
        self.val_labels = np.array(val_data[:,-1])
    
    def __getitem__(self, index):
        return self.val_data[index], self.val_labels[index]
    
    def __len__(self) -> int:
        return len(self.val_labels)

# # split train and val dataset
# def classwise_split(dataset_targets, ratio, num_classes):
#     # select {ratio*len(labels)} images from the images
#     train_val_label = np.array(dataset_targets)
#     train_indexes = []
#     val_indexes = []

#     for id in range(num_classes):
#         indexes = np.where(train_val_label == id)[0]
#         # print(len(indexes))
#         np.random.shuffle(indexes)
#         train_num = int(len(indexes)*ratio)
#         train_indexes.extend(indexes[:train_num])
#         val_indexes.extend(indexes[train_num:])
    
#     np.random.shuffle(train_indexes)
#     np.random.shuffle(val_indexes)

#     return train_indexes, val_indexes

# data_dir = "/data2/wyh-dataset/tabular-dataset"

# dataset = Adults(data_dir=data_dir)
# train_val_data = dataset.get_train_dataset()
# test_data = dataset.get_test_dataset()

# num_classes = 2
# train_targets = np.array(train_val_data.values[:,-1])
# train_indexes, val_indexes = classwise_split(train_targets, ratio=0.9, num_classes=2)

# train_data = np.array(train_val_data)[train_indexes]
# val_data = np.array(train_val_data)[val_indexes]
# test_data = np.array(test_data)

# train_dataset = Adults_train(train_data=train_data, noise_ratio=0.2)
# val_dataset = Adults_val(val_data=val_data)
# test_dataset = Adults_val(val_data=test_data)


# print(len(train_dataset))
# train_dataset.symmetric_noise()
# print(train_dataset.noise_or_not.count(True)) #6512
