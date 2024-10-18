import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import random_split
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from dataset.Clothing1M import Clothing1M_dataset
from dataset.Cifar10 import CIFAR10_Train, Cifar10_Val
from dataset.Mnist import MNIST_Train, MNIST_Val
from dataset.KMnist import KMNIST_Train, KMNIST_Val
from dataset.Svhn import SVHN_Train, SVHN_Val
from dataset.Cifar100 import CIFAR100_Train, Cifar100_Val
from dataset.Food101 import FOOD101_Train, FOOD101_Val
from dataset.Adults import Adults, Adults_train, Adults_val
from dataset.Convertype import load_covertype_dataset, Convertype_train, Convertype_val
# from dataset.DryBean import DryBeanDataset, DryBeanTrainDataset, DryBeanValDataset
# from dataset.Fetch_20news import fetch_20News, fetch_20News_Train, fetch_20News_Val
# from dataset.Imdb import Imdb, Imdb_train, Imdb_val
# from dataset.Hepmass import HepmassDataset, Hepmass_train, Hepmass_val

'''
    获取dataset(is_noise)的工具函数
'''
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

    
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)

    return train_indexes, val_indexes

# make noisy label
def label_noise(train_dataset, noise_type):
    if noise_type == 'sym_noise':
        train_dataset.symmetric_noise() 
    elif noise_type == 'asymmetric_noise':
        train_dataset.asymmetric_noise()

    return train_dataset

# get clean dataset according to args
# def get_dataset(args):
#     if args.dataset == "cifar10": #5w train_val /1w test 每一分类数量相等
#         from torchvision.datasets import CIFAR10
#         dataset = CIFAR10(root=args.data_root, train=True, download=False, transform=ToTensor())
#         test_dataset = CIFAR10(root=args.data_root, train=False, download=False, transform=ToTensor())
#         num_classes = 10

#     elif args.dataset == "mnist": #6w train_val / 1w test 每一分类数量不相等
#         from torchvision.datasets import MNIST
#         dataset = MNIST(root=args.data_root, train=True, download=False, transform=ToTensor())
#         test_dataset = MNIST(root=args.data_root, train=False, download=False, transform=ToTensor())
#         num_classes = 10

#     elif args.dataset == "kmnist": #6w train_val / 1w test 每一分类数量相等
#         from torchvision.datasets import KMNIST
#         dataset = KMNIST(root=args.data_root, train=True, download=False, transform=ToTensor())
#         test_dataset = KMNIST(root=args.data_root, train=False, download=False, transform=ToTensor())
#         num_classes = 10

#     elif args.dataset == "svhn": #73257 train_val / 26032 test 每一分类数量不想等 注意调取是dataset.labels
#         from torchvision.datasets import SVHN
#         data_root = args.data_root + "/SVHN"
#         dataset = SVHN(root=data_root, split="train", download=False, transform=ToTensor())
#         test_dataset = SVHN(root=data_root, split="test", download=False, transform=ToTensor())
#         num_classes = 10

#     elif args.dataset == "cifar100": #5w train_val /1w test 每一分类数量相等
#         from torchvision.datasets import CIFAR100
#         dataset = CIFAR100(root=args.data_root, train=True, download=False, transform=ToTensor())
#         test_dataset = CIFAR100(root=args.data_root, train=False, download=False, transform=ToTensor())
#         num_classes = 100

#     elif args.dataset == "food101": #75750 train_val /25250 test 每一分类数量相等
#         from torchvision.datasets import Food101
#         data_root = args.data_root + "/Food101"
#         input_size = 224

#         transform = transforms.Compose([
#             transforms.RandomResizedCrop(input_size),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ])
#         dataset = Food101(root=data_root, split="train", download=False, transform=transform)
#         test_dataset = Food101(root=data_root, split="test", download=False, transform=transform)
#         num_classes = 101
    
#     elif args.dataset == "clothing-1m": # train 265664 val 14313 test 10526
#         transform_train = transforms.Compose([
#             transforms.Resize(256),
#             transforms.RandomCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
#         ])
#         transform_test = transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
#         ])
#         data_root = args.data_root + "/Clothing1M"
#         train_dataset =  Clothing1M_dataset(data_root,transform=transform_train, mode='all')
#         val_dataset = Clothing1M_dataset(data_root,transform=transform_test, mode='val')
#         test_dataset =  Clothing1M_dataset(data_root,transform=transform_test, mode='test')
#         num_classes = 14
#         # 此处不需要继续划分train和val，因为clothing-1M已经提供val单独数据集
#         return train_dataset, val_dataset, test_dataset, num_classes
    
#     elif args.dataset == "adults":
#         # data_dir = "/data2/wyh-dataset/tabular-dataset"
#         dataset = Adults(data_dir=args.data_root)
#         train_val_data = dataset.get_train_dataset()
#         test_data = dataset.get_test_dataset()

#         num_classes = 2
#         train_targets = np.array(train_val_data.values[:,-1])
#         train_indexes, val_indexes = classwise_split(train_targets, ratio=args.train_ratio, num_classes=2)

#         train_data = np.array(train_val_data)[train_indexes]
#         val_data = np.array(train_val_data)[val_indexes]
#         test_data = np.array(test_data)

#         train_dataset = Adults_train(train_data=train_data, noise_ratio=0)
#         val_dataset = Adults_val(val_data=val_data)
#         test_dataset = Adults_val(val_data=test_data)

#         return train_dataset, val_dataset, test_dataset, num_classes
    
#     elif args.dataset == "convertype":
#         # data_dir = "/data2/wyh-dataset/tabular-dataset"
#         data, labels = load_covertype_dataset(args.data_root)
        
#         # split dataset 
#         num_classes = 7
#         train_val_indexes, test_indexes = classwise_split(dataset_targets=labels, ratio=0.8, num_classes=num_classes)
#         train_val_data = np.array(data)[train_val_indexes]
#         train_val_labels = np.array(labels)[train_val_indexes]
#         train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=args.train_ratio, num_classes=num_classes)
        
#         train_dataset = Convertype_train(data=train_val_data, labels=train_val_labels, train_indexes=train_indexes, noise_ratio=0)
#         val_dataset = Convertype_val(data=train_val_data, labels=train_val_labels,val_indexes=val_indexes)
#         test_dataset = Convertype_val(data=data, labels=labels, val_indexes=test_indexes)
        
#         return train_dataset, val_dataset, test_dataset, num_classes

#     # elif args.dataset == "drybean":
#     #     # data_dir = "/data2/wyh-dataset/tabular-dataset"
#     #     dataset = DryBeanDataset(data_dir=args.data_root)

#     #     # split dataset 
#     #     num_classes = 7
#     #     train_val_indexes, test_indexes = classwise_split(dataset_targets=dataset.labels, ratio=0.8, num_classes=num_classes)
#     #     train_val_data = np.array(dataset.data)[train_val_indexes]
#     #     train_val_labels = np.array(dataset.labels)[train_val_indexes]
#     #     train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=args.train_ratio, num_classes=num_classes)

#     #     train_dataset = DryBeanTrainDataset(data=train_val_data, labels=train_val_labels ,noise_ratio=0, train_indexes=train_indexes)
#     #     val_dataset = DryBeanValDataset(data = train_val_data, labels=train_val_labels, val_indexes=val_indexes)
#     #     test_dataset = DryBeanValDataset(data = dataset.data, labels= dataset.labels, val_indexes=test_indexes)

#     #     return train_dataset, val_dataset, test_dataset, num_classes
    
#     # elif args.dataset == "news":
#     #     # data_home="/data2/wyh-dataset/text-dataset/News"
#     #     dataset = fetch_20News()

#     #     # split dataset
#     #     num_classes = 20
#     #     train_val_data, train_val_labels = dataset.get_train_dataset()
#     #     test_data, test_labels = dataset.get_test_dataset()
#     #     bert_model = dataset.get_bert()
        
#     #     train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=args.train_ratio, num_classes=num_classes)
#     #     train_data = np.array(train_val_data)[train_indexes]
#     #     train_labels = np.array(train_val_labels)[train_indexes]
        
#     #     val_data = np.array(train_val_data)[val_indexes]
#     #     val_labels = np.array(train_val_labels)[val_indexes]

#     #     # load dataset
#     #     train_dataset = fetch_20News_Train(train_data=train_data, train_labels=train_labels, noise_ratio=0, bert_model = bert_model)
#     #     val_dataset = fetch_20News_Val(val_data=val_data, val_labels=val_labels, bert_model=bert_model)
#     #     test_dataset = fetch_20News_Val(val_data=test_data, val_labels=test_labels, bert_model=bert_model)

#     #     return train_dataset, val_dataset, test_dataset, num_classes
    
#     # elif args.dataset == "imdb":
#     #     imdb_dataset = Imdb(train_ratio = args.train_ratio)
        
#     #     num_classes = 2
#     #     bert_model = imdb_dataset.get_bert()
#     #     train_data = imdb_dataset.get_train_dataset()
#     #     val_data = imdb_dataset.get_val_dataset()
#     #     test_data = imdb_dataset.get_test_dataset()
#     #     train_dataset = Imdb_train(train_data=train_data, bert_model=bert_model, noise_ratio=0)
#     #     val_dataset = Imdb_val(val_data=val_data, bert_model=bert_model)
#     #     test_dataset = Imdb_val(val_data=test_data, bert_model=bert_model)

#     #     return train_dataset, val_dataset, test_dataset, num_classes
    
#     # elif args.dataset == "hepmass":
#     #     dataset = HepmassDataset(data_dir=args.data_root,train_ratio=args.train_ratio)
        
#     #     num_classes = 2
#     #     train_data, train_labels = dataset.get_train_dataset()
#     #     val_data, val_labels = dataset.get_val_dataset()
#     #     test_data, test_labels = dataset.get_test_dataset()

#     #     train_dataset = Hepmass_train(train_data=train_data, train_labels=train_labels, noise_ratio=0)
#     #     val_dataset = Hepmass_val(val_data=val_data, val_labels=val_labels)
#     #     test_dataset = Hepmass_val(val_data=test_data, val_labels=val_labels)

#     #     return train_dataset, val_dataset, test_dataset, num_classes
    
#     else:
#         raise ValueError("Invalid dataset parameter. Please choose a valid dataset.")
    
#     # 划分数据集 因为以上分类都是相等数量的，为方便直接使用random_split验证调用方法是否正确
#     train_size = int(len(dataset) * args.train_ratio)
#     val_size = int(len(dataset) - train_size)
#     train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    
#     if train_dataset is None or val_dataset is None or test_dataset is None:
#         raise ValueError("Dataset split failed. Please check the split ratios.")
    
#     return train_dataset, val_dataset, test_dataset, num_classes

def get_noisy_dataset(args):

    if args.dataset == "cifar10": #5w train_val /1w test 每一分类数量相等
        from torchvision.datasets import CIFAR10
        # load dataset
        dataset = CIFAR10(root=args.data_root, train=True, download=False, transform=ToTensor())
        test_dataset = CIFAR10(root=args.data_root, train=False, download=False, transform=ToTensor())
        
        # split dataset
        train_indexes, val_indexes = classwise_split(dataset_targets=dataset.targets, ratio=args.train_ratio, num_classes=10, coreset_size = args.coreset_size)
        
        # build train_dataset and add noise
        train_dataset = CIFAR10_Train(train_root=args.data_root, train_indexes=train_indexes, train=True, noise_ratio=args.noise_rate, transform=ToTensor())
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = Cifar10_Val(val_root=args.data_root, val_indexes=val_indexes, train=True, transform=ToTensor())
        
        # num_classes
        num_classes = 10
        return train_dataset, val_dataset, test_dataset, num_classes

    elif args.dataset == "mnist": #6w train_val / 1w test 每一分类数量不相等
        from torchvision.datasets import MNIST
        # load dataset
        dataset = MNIST(root=args.data_root, train=True, download=False, transform=ToTensor())
        test_dataset = MNIST(root=args.data_root, train=False, download=False, transform=ToTensor())
        
        # split dataset
        train_indexes, val_indexes = classwise_split(dataset_targets=dataset.targets, ratio=args.train_ratio, num_classes=10, coreset_size = args.coreset_size)
        
        # build train_dataset and add noise
        train_dataset = MNIST_Train(train_root=args.data_root, train_indexes=train_indexes, train=True, noise_ratio=args.noise_rate, transform=ToTensor())
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = MNIST_Val(val_root=args.data_root, val_indexes=val_indexes, train=True, transform=ToTensor())
        
        # num_classes
        num_classes = 10
        return train_dataset, val_dataset, test_dataset, num_classes

    elif args.dataset == "kmnist": #6w train_val / 1w test 每一分类数量相等
        from torchvision.datasets import KMNIST
        # load dataset
        dataset = KMNIST(root=args.data_root, train=True, download=False, transform=ToTensor())
        test_dataset = KMNIST(root=args.data_root, train=False, download=False, transform=ToTensor())
        
        # split dataset
        train_indexes, val_indexes = classwise_split(dataset_targets=dataset.targets, ratio=args.train_ratio, num_classes=10)
        
        # build train_dataset and add noise
        train_dataset = KMNIST_Train(train_root=args.data_root, train_indexes=train_indexes, train=True, noise_ratio=args.noise_rate, transform=ToTensor())
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = KMNIST_Val(val_root=args.data_root, val_indexes=val_indexes, train=True, transform=ToTensor())
        
        # num_classes
        num_classes = 10
        return train_dataset, val_dataset, test_dataset, num_classes
    
    elif args.dataset == "svhn": #73257 train_val / 26032 test 每一分类数量不想等 注意调取是dataset.labels
        from torchvision.datasets import SVHN
        data_root = args.data_root + "/SVHN"
        # load dataset
        dataset = SVHN(root=data_root, split="train", download=False, transform=ToTensor())
        test_dataset = SVHN(root=data_root, split="test", download=False, transform=ToTensor())
        
        # split dataset
        train_indexes, val_indexes = classwise_split(dataset_targets=dataset.labels, ratio=args.train_ratio, num_classes=10, coreset_size = args.coreset_size)

        # build train_dataset and add noise
        train_dataset = SVHN_Train(train_root=data_root, train_indexes=train_indexes, split="train", noise_ratio=args.noise_rate, transform=ToTensor())
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = SVHN_Val(val_root=data_root, val_indexes=val_indexes, split="train", transform=ToTensor())
        
        # num_classes
        num_classes = 10
        return train_dataset, val_dataset, test_dataset, num_classes
    
    elif args.dataset == "cifar100": #5w train_val /1w test 每一分类数量相等
        from torchvision.datasets import CIFAR100
        # load dataset
        dataset = CIFAR100(root=args.data_root, train=True, download=False, transform=ToTensor())
        test_dataset = CIFAR100(root=args.data_root, train=False, download=False, transform=ToTensor())
        
        # split dataset
        train_indexes, val_indexes = classwise_split(dataset_targets=dataset.targets, ratio=args.train_ratio, num_classes=100, coreset_size = args.coreset_size)
        
        # build train_dataset and add noise
        train_dataset = CIFAR100_Train(train_root=args.data_root, train_indexes=train_indexes, train=True, noise_ratio=args.noise_rate, transform=ToTensor())
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = Cifar100_Val(val_root=args.data_root, val_indexes=val_indexes, train=True, transform=ToTensor())
        
        # num_classes
        num_classes = 100
        return train_dataset, val_dataset, test_dataset, num_classes
    
    elif args.dataset == "food101": #75750 train_val /25250 test 每一分类数量相等
        from torchvision.datasets import Food101
        data_root = args.data_root + "/Food101"
        input_size = 224

        transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        # load dataset
        dataset = Food101(root=data_root, split="train", download=False, transform=transform)
        test_dataset = Food101(root=data_root, split="test", download=False, transform=transform)
        
        # split dataset
        train_indexes, val_indexes = classwise_split(dataset_targets=dataset._labels, ratio=args.train_ratio, num_classes=101, coreset_size = args.coreset_size)
        
        # build train_dataset and add noise
        train_dataset = FOOD101_Train(train_root=data_root, train_indexes=train_indexes, split="train", noise_ratio=args.noise_rate, transform=transform)
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = FOOD101_Val(val_root=data_root, val_indexes=val_indexes, split="train", transform=transform)
        
        # num_classes
        num_classes = 101
        return train_dataset, val_dataset, test_dataset, num_classes
    
    elif args.dataset == "clothing-1m": # train 265664 val 14313 test 10526
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.6959, 0.6537, 0.6371), (0.3113, 0.3192, 0.3214)),
        ])
        data_root = args.data_root + "/Clothing1M"
        train_dataset =  Clothing1M_dataset(data_root,transform=transform_train, mode='all')
        val_dataset = Clothing1M_dataset(data_root,transform=transform_test, mode='val')
        test_dataset =  Clothing1M_dataset(data_root,transform=transform_test, mode='test')
        num_classes = 14
        
        return train_dataset, val_dataset, test_dataset, num_classes
    
    elif args.dataset == "adults":
        # data_dir = "/data2/wyh-dataset/tabular-dataset"
        dataset = Adults(data_dir=args.data_root)
        train_val_data = dataset.get_train_dataset()
        test_data = dataset.get_test_dataset()

        num_classes = 2
        train_targets = np.array(train_val_data.values[:,-1])
        train_indexes, val_indexes = classwise_split(train_targets, ratio=args.train_ratio, num_classes=num_classes)

        train_data = np.array(train_val_data)[train_indexes]
        val_data = np.array(train_val_data)[val_indexes]
        test_data = np.array(test_data)

        train_dataset = Adults_train(train_data=train_data, noise_ratio=args.noise_rate)
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = Adults_val(val_data=val_data)
        test_dataset = Adults_val(val_data=test_data)

        return train_dataset, val_dataset, test_dataset, num_classes
    
    elif args.dataset == "convertype":
        # data_dir = "/data2/wyh-dataset/tabular-dataset"
        data, labels = load_covertype_dataset(args.data_root)
        
        # split dataset 
        num_classes = 7
        train_val_indexes, test_indexes = classwise_split(dataset_targets=labels, ratio=0.8, num_classes=num_classes)
        train_val_data = np.array(data)[train_val_indexes]
        train_val_labels = np.array(labels)[train_val_indexes]
        train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=args.train_ratio, num_classes=num_classes)
        
        train_dataset = Convertype_train(data=train_val_data, labels=train_val_labels, train_indexes=train_indexes, noise_ratio=args.noise_rate)
        train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
        val_dataset = Convertype_val(data=train_val_data, labels=train_val_labels,val_indexes=val_indexes)
        test_dataset = Convertype_val(data=data, labels=labels, val_indexes=test_indexes)
        
        return train_dataset, val_dataset, test_dataset, num_classes
    
    # elif args.dataset == "drybean":
    #     # data_dir = "/data2/wyh-dataset/tabular-dataset"
    #     dataset = DryBeanDataset(data_dir=args.data_root)

    #     # split dataset 
    #     num_classes = 7
    #     train_val_indexes, test_indexes = classwise_split(dataset_targets=dataset.labels, ratio=0.8, num_classes=num_classes)
    #     train_val_data = np.array(dataset.data)[train_val_indexes]
    #     train_val_labels = np.array(dataset.labels)[train_val_indexes]
    #     train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=args.train_ratio, num_classes=num_classes)

    #     train_dataset = DryBeanTrainDataset(data=train_val_data, labels=train_val_labels ,noise_ratio=args.noise_rate, train_indexes=train_indexes)
    #     train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
    #     val_dataset = DryBeanValDataset(data = train_val_data, labels=train_val_labels, val_indexes=val_indexes)
    #     test_dataset = DryBeanValDataset(data = dataset.data, labels= dataset.labels, val_indexes=test_indexes)

    #     return train_dataset, val_dataset, test_dataset, num_classes

    # elif args.dataset == "news":
    #     dataset = fetch_20News()
    #     # split dataset
    #     num_classes = 20
    #     train_val_data, train_val_labels = dataset.get_train_dataset()
    #     test_data, test_labels = dataset.get_test_dataset()
    #     bert_model = dataset.get_bert()
        
    #     train_indexes, val_indexes = classwise_split(dataset_targets=train_val_labels, ratio=args.train_ratio, num_classes=num_classes)
        
    #     train_data = np.array(train_val_data, dtype=object)[train_indexes]
    #     train_labels = np.array(train_val_labels)[train_indexes]
        
    #     val_data = np.array(train_val_data, dtype=object)[val_indexes]
    #     val_labels = np.array(train_val_labels)[val_indexes]

    #     # load dataset
    #     train_dataset = fetch_20News_Train(train_data=train_data, train_labels=train_labels, noise_ratio=args.noise_rate, bert_model = bert_model)
    #     train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
    #     val_dataset = fetch_20News_Val(val_data=val_data, val_labels=val_labels, bert_model=bert_model)
    #     test_dataset = fetch_20News_Val(val_data=test_data, val_labels=test_labels, bert_model=bert_model)

    #     return train_dataset, val_dataset, test_dataset, num_classes
    
    # elif args.dataset == "imdb":
    #     imdb_dataset = Imdb(train_ratio = args.train_ratio)
        
    #     num_classes = 2
    #     bert_model = imdb_dataset.get_bert()
    #     train_data = imdb_dataset.get_train_dataset()
    #     val_data = imdb_dataset.get_val_dataset()
    #     test_data = imdb_dataset.get_test_dataset()
        
    #     train_dataset = Imdb_train(train_data=train_data, bert_model=bert_model, noise_ratio=args.noise_rate)
    #     train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
    #     val_dataset = Imdb_val(val_data=val_data, bert_model=bert_model)
    #     test_dataset = Imdb_val(val_data=test_data, bert_model=bert_model)

    #     return train_dataset, val_dataset, test_dataset, num_classes
    
    # elif args.dataset == "hepmass":
    #     dataset = HepmassDataset(data_dir=args.data_root,train_ratio=args.train_ratio)
        
    #     num_classes = 2
    #     train_data, train_labels = dataset.get_train_dataset()
    #     val_data, val_labels = dataset.get_val_dataset()
    #     test_data, test_labels = dataset.get_test_dataset()

    #     train_dataset = Hepmass_train(train_data=train_data, train_labels=train_labels, noise_ratio=args.noise_rate)
    #     train_dataset = label_noise(train_dataset=train_dataset, noise_type=args.noise_type)
    #     val_dataset = Hepmass_val(val_data=val_data, val_labels=val_labels)
    #     test_dataset = Hepmass_val(val_data=test_data, val_labels=val_labels)
        
    #     return train_dataset, val_dataset, test_dataset, num_classes

'''
    根据dataset获取模型需要的参数
''' 
def get_dataset_params(args):
    if args.dataset == 'mnist':
        input_channel = 1
        top_bn = False
        epoch_decay_start = 80
        n_epoch = 50
    elif args.dataset == 'cifar10':
        input_channel = 3
        top_bn = False
        epoch_decay_start = 80
        n_epoch = 50
    elif args.dataset == 'cifar100':
        input_channel = 3
        top_bn = False
        epoch_decay_start = 100
        n_epoch = 2
    elif args.dataset == 'colthing-1m':
        input_channel = 3
        top_bn = False
        epoch_decay_start = 100
        n_epoch = 2
    else:
        input_channel = 3
        top_bn = False
        epoch_decay_start = 100
        n_epoch = 50
        # raise ValueError(f"Invalid dataset: {args.dataset}")
    
    if args.forget_rate is None:
        forget_rate=args.noise_rate
    else:
        forget_rate=args.forget_rate

    return input_channel, forget_rate, top_bn, epoch_decay_start, n_epoch

'''
    coteaching方法所需要的优化器动态参数工具函数
'''
# def get_optimizer_plan(args):
#     mom1 = 0.9
#     mom2 = 0.1
#     alpha_plan = [args.lr] * args.n_epoch
#     beta1_plan = [mom1] * args.n_epoch
#     for i in range(args.epoch_decay_start, args.n_epoch):
#         alpha_plan[i] = float(args.n_epoch - i) / (args.n_epoch - args.epoch_decay_start) * args.lr
#         beta1_plan[i] = mom2
    
#     return alpha_plan, beta1_plan
    

# def adjust_learning_rate(optimizer, epoch, alpha_plan, beta1_plan):
#     for param_group in optimizer.param_groups:
#         param_group['lr']=alpha_plan[epoch]
#         param_group['betas']=(beta1_plan[epoch], 0.999) # Only change beta1

# def get_rate_schedule(args, forget_rate):
#     rate_schedule = np.ones(args.n_epoch)*forget_rate
#     rate_schedule[:args.num_gradual] = np.linspace(0, forget_rate**args.exponent, args.num_gradual)
#     return rate_schedule

# '''
#     训练模型过程的调用工具函数
# '''
# def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
#     loss_1 = F.cross_entropy(y_1, t, reduction='none')
#     loss_2 = F.cross_entropy(y_2, t, reduction='none')

#     loss_1_np = loss_1.detach().cpu().numpy()
#     loss_2_np = loss_2.detach().cpu().numpy()

#     ind_1_sorted = np.argsort(loss_1_np)
#     ind_2_sorted = np.argsort(loss_2_np)

#     loss_1_sorted = loss_1[ind_1_sorted]
#     loss_2_sorted = loss_2[ind_2_sorted]

#     remember_rate = 1 - forget_rate
#     num_remember = int(remember_rate * len(loss_1_sorted))

#     pure_ratio_1 = np.sum(np.array(noise_or_not)[ind[ind_1_sorted[:num_remember]]]) / float(num_remember)
#     pure_ratio_2 = np.sum(np.array(noise_or_not)[ind[ind_2_sorted[:num_remember]]]) / float(num_remember)

#     ind_1_update = ind_1_sorted[:num_remember]
#     ind_2_update = ind_2_sorted[:num_remember]
    
#     # 将相关张量移回CUDA设备
#     ind_1_update = torch.from_numpy(ind_1_update).cuda()
#     ind_2_update = torch.from_numpy(ind_2_update).cuda()
#     loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
#     loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

#     return torch.sum(loss_1_update) / num_remember, torch.sum(loss_2_update) / num_remember, pure_ratio_1, pure_ratio_2

# def accuracy(logit, target, topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     output = F.softmax(logit, dim=1)
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.view(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


# def train(args, train_loader, epoch, model1, optimizer1, model2, optimizer2, rate_schedule, noise_or_not, traindataset_size):
#     pure_ratio_1_list=[]
#     pure_ratio_2_list=[]
    
#     train_total=0
#     train_correct=0 
#     train_total2=0
#     train_correct2=0 

#     for i, (images, labels, _, _, _, indexes) in enumerate(train_loader):
#         ind=indexes.cpu().numpy().transpose()
#         if i>args.num_iter_per_epoch:
#             break
      
#         images = Variable(images).cuda()
#         labels = Variable(labels).cuda()
        
#         # Forward + Backward + Optimize
#         logits1=model1(images)
#         prec1, _ = accuracy(logits1, labels, topk=(1, 5))
#         train_total+=1
#         train_correct+=prec1

#         logits2 = model2(images)
#         prec2, _ = accuracy(logits2, labels, topk=(1, 5))
#         train_total2+=1
#         train_correct2+=prec2
#         loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(logits1, logits2, labels, rate_schedule[epoch], ind, noise_or_not)
#         pure_ratio_1_list.append(100*pure_ratio_1)
#         pure_ratio_2_list.append(100*pure_ratio_2)

#         optimizer1.zero_grad()
#         loss_1.backward()
#         optimizer1.step()
#         optimizer2.zero_grad()
#         loss_2.backward()
#         optimizer2.step()
#         if (i+1) % args.print_freq == 0:
#             print ('Epoch [%d/%d], Iter [%d/%d] Training Accuracy1: %.4F, Training Accuracy2: %.4f, Loss1: %.4f, Loss2: %.4f, Pure Ratio1: %.4f, Pure Ratio2 %.4f' 
#                   %(epoch+1, args.n_epoch, i+1, traindataset_size//args.train_batch_size, prec1, prec2, loss_1.item(), loss_2.item(), np.sum(pure_ratio_1_list)/len(pure_ratio_1_list), np.sum(pure_ratio_2_list)/len(pure_ratio_2_list)))

#     train_acc1=float(train_correct)/float(train_total)
#     train_acc2=float(train_correct2)/float(train_total2)
#     return train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list

# # Evaluate the Model
# def evaluate(test_loader, model1, model2):
#     model1.eval()    # Change model to 'eval' mode.
#     correct1 = 0
#     total1 = 0
#     for images, labels in test_loader:
#         images = Variable(images).cuda()
#         logits1 = model1(images)
#         outputs1 = F.softmax(logits1, dim=1)
#         _, pred1 = torch.max(outputs1.data, 1)
#         total1 += labels.size(0)
#         correct1 += (pred1.cpu() == labels).sum()

#     model2.eval()    # Change model to 'eval' mode 
#     correct2 = 0
#     total2 = 0
#     for images, labels in test_loader:
#         images = Variable(images).cuda()
#         logits2 = model2(images)
#         outputs2 = F.softmax(logits2, dim=1)
#         _, pred2 = torch.max(outputs2.data, 1)
#         total2 += labels.size(0)
#         correct2 += (pred2.cpu() == labels).sum()
 
#     acc1 = 100*float(correct1)/float(total1)
#     acc2 = 100*float(correct2)/float(total2)
#     return acc1, acc2