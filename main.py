import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import argparse, sys
import numpy as np
import datetime
import shutil

# IMPORT DATASET AND LOSS FROM UTILS\
from parser.get_parser import get_parser
from utils import *

# IMPORT MODELS FROM MODEL
from model.cnn import CNN
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # load dataset 已经完成图像数据集 对称非对称部分的代码任务
    train_dataset, val_dataset, test_dataset, num_classes = get_noisy_dataset(args)
    
    print(train_dataset[0])
    # # load params 进行coteaching方法获取模型训练参数
    # input_channel, forget_rate, args.top_bn, args.epoch_decay_start, args.n_epoch = get_dataset_params(args)
    
    # # Adjust learning rate and betas for Adam Optimizer 动态调整优化器参数
    # alpha_plan, beta1_plan = get_optimizer_plan(args=args)

    # # define drop rate schedule 设置训练时的数据集动态丢失率
    # rate_schedule = get_rate_schedule(args=args, forget_rate=forget_rate)

    # # save result 多方法则调用args.method
    # save_dir = args.result_dir +'/' +args.dataset+'/coteaching/'
    # if not os.path.exists(save_dir):
    #     os.system('mkdir -p %s' % save_dir)
    
    # model_str=args.dataset+'_coteaching_'+args.noise_type+'_'+str(args.noise_rate)
    # txtfile=save_dir+"/"+model_str+".txt"
    # nowTime=datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    # if os.path.exists(txtfile):
    #     os.system('mv %s %s' % (txtfile, txtfile+".bak-%s" % nowTime))
    
    # # data loader 注意这里去掉最后一个（可能未满）的batch
    # print(f"loading dataset...{args.dataset}")
    # noise_or_not = train_dataset.noise_or_not
    # train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, drop_last=True, num_workers=args.num_workers, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.test_batch_size, drop_last=True, num_workers=args.num_workers, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, drop_last=True, num_workers=args.num_workers, shuffle=False)

    # # define model 原论文是选择原始的两个CNN这里我们先模仿，后续添加更多...
    # print(f"building model...{args.model}")
    # model1 = CNN(input_channel=input_channel, n_outputs=num_classes)
    # model1.cuda()
    # print(model1.parameters)
    # optimizer1 = torch.optim.Adam(model1.parameters(), lr=args.lr)

    # model2 = CNN(input_channel=input_channel, n_outputs=num_classes)
    # model2.cuda()
    # print(model2.parameters)
    # optimizer2 = torch.optim.Adam(model2.parameters(), lr=args.lr)

    # with open(txtfile, "a") as myfile:
    #     myfile.write('epoch: train_acc1 train_acc2 test_acc1 test_acc2 pure_ratio1 pure_ratio2\n')

    # mean_pure_ratio1 = 0
    # mean_pure_ratio2 = 0
    
    # epoch=0
    # train_acc1=0
    # train_acc2=0
    # # evaluate models with random weights
    # test_acc1, test_acc2 = evaluate(test_loader, model1, model2)
    # print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %% Pure Ratio1 %.4f %% Pure Ratio2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    # # save results
    # with open(txtfile, "a") as myfile:
    #     myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' '  + str(mean_pure_ratio1) + ' '  + str(mean_pure_ratio2) + "\n")

    # # training
    # for epoch in range(1, args.n_epoch):
    #     # train models
    #     model1.train()
    #     adjust_learning_rate(optimizer1, epoch, alpha_plan, beta1_plan)
    #     model1.train()
    #     adjust_learning_rate(optimizer2, epoch, alpha_plan, beta1_plan)

    #     print (f"Training f{model_str}...")
    #     train_acc1, train_acc2, pure_ratio_1_list, pure_ratio_2_list = train(args, train_loader, epoch, model1, optimizer1, model2, optimizer2, rate_schedule, noise_or_not, len(train_dataset))
    #     # evaluate models
    #     print (f"Evaluatinging f{model_str}...")
    #     test_acc1, test_acc2 = evaluate(test_loader, model1, model2)
    #     # save results
    #     mean_pure_ratio1 = sum(pure_ratio_1_list)/len(pure_ratio_1_list)
    #     mean_pure_ratio2 = sum(pure_ratio_2_list)/len(pure_ratio_2_list)
    #     print('Epoch [%d/%d] Test Accuracy on the %s test images: Model1 %.4f %% Model2 %.4f %%, Pure Ratio 1 %.4f %%, Pure Ratio 2 %.4f %%' % (epoch+1, args.n_epoch, len(test_dataset), test_acc1, test_acc2, mean_pure_ratio1, mean_pure_ratio2))
    #     with open(txtfile, "a") as myfile:
    #         myfile.write(str(int(epoch)) + ': '  + str(train_acc1) +' '  + str(train_acc2) +' '  + str(test_acc1) + " " + str(test_acc2) + ' ' + str(mean_pure_ratio1) + ' ' + str(mean_pure_ratio2) + "\n")

    

    

