# -*- coding: utf-8 -*_
import torch
import torch.nn as nn
import torch.optim as optim
# Bert model and its tokenizer
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
# Text data
from torchtext import data, datasets
import numpy as np
import random
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Imdb():
    def __init__(self, train_ratio, data_dir = '/data2/wyh-dataset/text-dataset/bert-base-uncase'):
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
        
        text = data.Field(
            batch_first=True,
            use_vocab=False,
            tokenize=tokenize_and_crop,
            preprocessing=self.tokenizer.convert_tokens_to_ids,
            init_token=init_token_id,
            pad_token=pad_token_id,
            unk_token=unk_token_id
        )
        label_map = {'pos': 0, 'neg': 1}
        label = data.LabelField(dtype=torch.float, 
                        preprocessing=lambda x: label_map[x])
        
        self.train_data, self.test_data  = datasets.IMDB.splits(text, label, root="/data2/wyh-dataset/text-dataset")
        self.train_data, self.valid_data = self.train_data.split(split_ratio=train_ratio)
        print(len(self.train_data))
    
    def get_train_dataset(self):
        return self.train_data
    
    def get_val_dataset(self):
        return self.valid_data
    
    def get_test_dataset(self):
        return self.test_data
    
    def get_bert(self):
        return self.bert_model


class Imdb_train(Dataset):
    def __init__(self, train_data, bert_model, noise_ratio):
        # define the number of classes
        self.num_classes = 2

        # define data
        self.train_data = list(train_data.text)
        self.true_labels = np.array(list(train_data.label))
        self.train_labels = np.array(list(train_data.label))
        self.soft_labels = np.zeros((len(self.train_labels), self.num_classes), dtype=np.float32)

        # define noise ratio
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

class Imdb_val(Dataset):
    def __init__(self, val_data, bert_model):
        # define the number of classes
        self.num_classes = 2

        # define data
        self.val_data = list(val_data.text)
        self.val_labels = list(val_data.label)
    
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

# imdb_dataset = Imdb()

# bert_model = imdb_dataset.get_bert()
# train_data = imdb_dataset.get_train_dataset()
# val_data = imdb_dataset.get_val_dataset()
# test_data = imdb_dataset.get_test_dataset()


# train_dataset = Imdb_train(train_data=train_data, bert_model=bert_model, noise_ratio=0.2)
# val_dataset = Imdb_val(val_data=val_data, bert_model=bert_model)
# test_dataset = Imdb_val(val_data=test_data, bert_model=bert_model)
# train_dataset.symmetric_noise()

# print(train_dataset.noise_or_not.count(True)) # 4500 对称 #4499非对称
  
