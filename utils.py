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

import torch
import shutil
import os
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def prepare_folders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best, filename=''):
    if not filename:
      filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

'''
    get dataset(is_noise)
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
        # data_dir = "tabular-dataset"
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
        # data_dir = "tabular-dataset"
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


def prepare_folders(args):
    """Create log and model folder"""
    folders_util = [args.root_log, args.root_model,
                    os.path.join(args.root_log, args.store_name),
                    os.path.join(args.root_model, args.store_name)]
    for folder in folders_util:
        if not os.path.exists(folder):
            print('creating folder ' + folder)
            os.mkdir(folder)

def save_checkpoint(args, state, is_best, filename=''):
    if not filename:
      filename = '%s/%s/ckpt.pth.tar' % (args.root_model, args.store_name)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    

def deem_accuracy(logit, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    output = F.softmax(logit, dim=1)
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def deem_train(train_loader, epoch, model, optimizer, criterion):
    
    train_total=0
    train_correct=0
    running_loss = 0.0

    #other datasets
    for i, (images, labels, _, _, _, indexes) in enumerate(train_loader):

    # # colthing-1m
    # for i, (images, labels) in enumerate(train_loader):  
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        logits = model(images)
        prec1, _ = accuracy(logits, labels, topk=(1, 5))
        train_total+=1
        train_correct+=prec1

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:  # 每 100 个 batch 输出一次
            print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.4f}")
            running_loss = 0.0

def deem_evaluate(test_loader, model):
    model.eval()    # Change model to "eval" mode.
    correct1 = 0
    total1 = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = Variable(images).cuda()
            logits1 = model(images)
            outputs1 = F.softmax(logits1, dim=1)
            _, pred1 = torch.max(outputs1.data, 1)
            total1 += labels.size(0)
            correct1 += (pred1.cpu() == labels).sum()

 
    acc1 = 100*float(correct1)/float(total1)
    return acc1
