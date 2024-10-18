import os
import torch
import copy
import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import fetch_20newsgroups
from transformers import BertTokenizer, BertModel
from torchtext import data, datasets

categories = ['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']

os.environ['CUDA_VISIBLE_DEVICES'] = '1' 
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class fetch_20News():
    def __init__(self, data_dir = '/data2/wyh-dataset/text-dataset/bert-base-uncase'):
        # 加载BERT模型和分词器
        self.bert_model = BertModel.from_pretrained(data_dir)  
        self.tokenizer = BertTokenizer.from_pretrained(data_dir)
        self.bert_model.to(device)
        # get token for bert
        init_token_id = self.tokenizer.cls_token_id
        eos_token_id  = self.tokenizer.sep_token_id
        pad_token_id  = self.tokenizer.pad_token_id
        unk_token_id  = self.tokenizer.unk_token_id

        max_input_len = self.tokenizer.max_model_input_sizes['bert-base-uncased']

        # Tokensize and crop sentence to 510 (for 1st and last token) instead of 512 (i.e. `max_input_len`)
        def tokenize_and_crop(sentence):
            tokens = self.tokenizer.tokenize(sentence)
            tokens = tokens[:max_input_len - 2]
            return tokens
        
        def preprocess_data(text_list):
            # 对列表中的每个字符串进行预处理操作
            processed_text_list = []
            for text in text_list:
                tokens = tokenize_and_crop(text)  # 调用 tokenize_and_crop 函数进行切分和裁剪处理
                token_ids = [init_token_id] + self.tokenizer.convert_tokens_to_ids(tokens) + [eos_token_id] # 添加起始 token 并转换为 token ID
                processed_text_list.append(token_ids)
            return processed_text_list
        
        train_dataset = fetch_20newsgroups(data_home="/data2/wyh-dataset/text-dataset/News", 
                                      download_if_missing=False,
                                      subset='train', shuffle=True, 
                                      categories=categories)
        test_dataset = fetch_20newsgroups(data_home="/data2/wyh-dataset/text-dataset/News", 
                                      download_if_missing=False,
                                      subset='test', shuffle=True, 
                                      categories=categories)
        
        self.train_data = preprocess_data(train_dataset.data)
        self.train_labels = train_dataset.target
        self.test_data = preprocess_data(test_dataset.data)
        self.test_labels = test_dataset.target
    
    
    def get_train_dataset(self):
        return self.train_data, self.train_labels
    
    def get_test_dataset(self):
        return self.test_data, self.test_labels
    
    def get_bert(self):
        return self.bert_model
    
class fetch_20News_Train(Dataset):
    def __init__(self, noise_ratio, train_data, train_labels, bert_model):
        # define the number of classes
        self.num_classes = 20

        # define data
        self.train_data = train_data
        self.true_labels = copy.deepcopy(train_labels)
        self.train_labels = copy.deepcopy(train_labels)
        self.soft_labels = np.zeros((len(self.train_labels), self.num_classes), dtype=np.float32)

        # define noise_ratio
        self.noise_ratio = noise_ratio
        self.noise_or_not = [False] * len(self.train_labels)  # 初始化为全部为False

        # define bert model 
        self.bert_model = bert_model
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
        dic = {0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9,
               9:10, 10:11, 11:12, 12:13, 13:14, 14:15, 15:16, 
               16:17, 17:18, 18:19, 19:0}

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
        text = self.train_data[index]
        train_label = self.train_labels[index]
        true_label = self.true_labels[index]

        # get bert model pooler_output
        tensor = torch.LongTensor(text).to(device)
        tensor = tensor.unsqueeze(0)
        output = self.bert_model(tensor)
        pooled_output = output.pooler_output
        pooled_output = pooled_output.squeeze(0)

        # 判断是否是噪音数据
        label = self.train_labels[index]
        true_label = self.true_labels[index]
        is_noise = False
        if label != true_label:
            is_noise = True
        
        return pooled_output, train_label, true_label, is_noise, self.soft_labels[index], index

class fetch_20News_Val():
    def __init__(self, val_data, val_labels, bert_model):
        # define the number of classes
        self.num_classes = 20

        # define data
        self.val_data = val_data
        self.val_labels = val_labels
        
        # define bert model 
        self.bert_model = bert_model
    
    def __getitem__(self, index):
        text = self.val_data[index]
        label = self.val_labels[index]

        # get bert model pooler_output
        tensor = torch.LongTensor(text).to(device)
        tensor = tensor.unsqueeze(0)
        output = self.bert_model(tensor)
        pooled_output = output.pooler_output
        pooled_output = pooled_output.squeeze(0)
        
        return pooled_output, label

# split train and val dataset
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

# dataset = fetch_20News()
# train_val_data, train_val_labels = dataset.get_train_dataset()
# test_data, test_labels = dataset.get_test_dataset()
# bert_model = dataset.get_bert()
# train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=0.9, num_classes=20)
# train_data = np.array(train_val_data)[train_indexes]
# train_labels = np.array(train_val_labels)[train_indexes]
# val_data = np.array(train_val_data)[val_indexes]
# val_labels = np.array(train_val_labels)[val_indexes]

# train_dataset = fetch_20News_Train(train_data=train_data, train_labels=train_labels, noise_ratio=0.2, bert_model = bert_model)
# val_dataset = fetch_20News_Val(val_data=val_data, val_labels=val_labels, bert_model=bert_model)
# test_dataset = fetch_20News_Val(val_data=test_data, val_labels=test_labels, bert_model=bert_model)

# train_dataset.asymmetric_noise()
# print(len(train_dataset.true_labels)) # 10174
# print(train_dataset.noise_or_not.count(True)) # 2043