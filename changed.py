import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import models_test as models
import math
import os
import sys
import time
import argparse
import datetime
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.data import DataLoader
import random
import os
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import csv
import torch.distributed as dist
from typing import Dict, List
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


parser = argparse.ArgumentParser(description='PyTorch datasets changed')
parser.add_argument('--lr', default=0.0008, type=float, help='learning_rate')
parser.add_argument('--meta_lr', default=0.02, type=float, help='meta learning_rate')
parser.add_argument('--num_fast', default=10, type=int, help='number of random perturbations')
parser.add_argument('--perturb_ratio', default=0.5, type=float, help='ratio of random perturbations')
parser.add_argument('--start_iter', default=500, type=int)
parser.add_argument('--mid_iter', default=2000, type=int)
parser.add_argument('--start_epoch', default=1, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--alpha', default=1, type=int)
parser.add_argument('--eps', default=0.99, type=float, help='Running average of model weights')
parser.add_argument('--seed', default=123)
parser.add_argument('--gpuid', default=0, type=int)
parser.add_argument('--id', default='Changed')
parser.add_argument('--checkpoint', default='cross_entropy')
args = parser.parse_args()

random.seed(args.seed)
torch.cuda.set_device(args.gpuid)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
use_cuda = torch.cuda.is_available()

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
# Training
def train(epoch):
    global init
    net.train()
    tch_net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate=learning_rate/10        
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    
    for batch_idx, sample in enumerate(train_loader):
        if use_cuda:
            inputs=sample['data']
            targets=sample['label']
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)               # Forward Propagation
        
        class_loss = criterion(outputs, targets)  # Loss
        class_loss.backward(retain_graph=True)  

        if batch_idx>args.start_iter or epoch>1:
            if batch_idx>args.mid_iter or epoch>1:
                args.eps=0.999
                alpha = args.alpha
            else:
                u = (batch_idx-args.start_iter)/(args.mid_iter-args.start_iter)
                alpha = args.alpha*math.exp(-5*(1-u)**2)          
          
            if init:
                init = False
                for param,param_tch in zip(net.parameters(),tch_net.parameters()): 
                    param_tch.data.copy_(param.data)                    
            else:
                for param,param_tch in zip(net.parameters(),tch_net.parameters()):
                    param_tch.data.mul_(args.eps).add_((1-args.eps), param.data)   
            
            _,feats = pretrain_net(inputs,get_feat=True)
            tch_outputs = tch_net(inputs,get_feat=False)
            p_tch = F.softmax(tch_outputs,dim=1)
            p_tch = p_tch.detach()
            
            for i in range(args.num_fast):
                targets_fast = targets.clone()
                randidx = torch.randperm(targets.size(0))
                for n in range(int(targets.size(0)*args.perturb_ratio)):
                    num_neighbor = 1
                    idx = randidx[n]
                    feat = feats[idx]
                    feat.view(1,feat.size(0))
                    feat.data = feat.data.expand(targets.size(0),feat.size(0))
                    dist = torch.sum((feat-feats)**2,dim=1)
                    _, neighbor = torch.topk(dist.data,num_neighbor+1,largest=False)
                    targets_fast[idx] = targets[neighbor[random.randint(1,num_neighbor)]]
                    
                fast_loss = criterion(outputs,targets_fast)

                grads = torch.autograd.grad(fast_loss, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
                for grad in grads:
                    grad = grad.detach()
                    grad.requires_grad = False  
   
                fast_weights = OrderedDict((name, param - args.meta_lr*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))
                
                fast_out , other_info = net.forward(inputs,fast_weights)  
                logp_fast = F.log_softmax(fast_out,dim=1)
        
                if i == 0:
                    consistent_loss = consistent_criterion(logp_fast,p_tch)
                else:
                    consistent_loss = consistent_loss + consistent_criterion(logp_fast,p_tch)
                
            meta_loss = consistent_loss*alpha/args.num_fast 
            
            meta_loss.backward()
                
        optimizer.step() # Optimizer update

        train_loss += class_loss.item()  
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, class_loss.item(), 100.*correct/total))
        sys.stdout.flush()
        if (batch_idx+1)%1000==0:
            val(epoch,batch_idx,train_loader,val_loader)
            val_tch(epoch,batch_idx)
            net.train()
            tch_net.train()            
            
##############################################################################################
from she_postprocessor import SHEPostprocessor
from base_postprocessor import BasePostprocessor
def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()
def is_main_process() -> bool:
    return get_rank() == 0
def postprocess( net: nn.Module, data: torch.Tensor):
        with torch.no_grad():
            output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf
def inference(
                net: nn.Module,
                data_loader: DataLoader,
                progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        # she_postprocessor=SHEPostprocessor()
        # she_postprocessor.setup(net, train_loader, val_loader)
        for sample in tqdm(data_loader,disable=not progress or not is_main_process()):
            data = sample['data'].cuda()
            label = sample['label'].cuda()
            with torch.no_grad():
                pred, conf = postprocess(net, data)
        
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list

def inference_val(
                net: nn.Module,
                data_loader: DataLoader,
                progress: bool = True):
        pred_list, conf_list, label_list = [], [], []
        # she_postprocessor=SHEPostprocessor()
        # she_postprocessor.setup(net, train_loader, val_loader)
        for data,label in tqdm(data_loader,disable=not progress or not is_main_process()):
            data = data.cuda()
            label = label.cuda()
            with torch.no_grad():
                pred, conf = postprocess(net, data)
        
            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)

    accuracy = acc(pred, label)

    results = [fpr, auroc, aupr_in, aupr_out, accuracy]

    return results
def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc
# auc
def auc_and_fpr_recall(conf, label, tpr_th):
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    conf = np.nan_to_num(conf)      

    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    precision_in, recall_in, thresholds_in \
        = metrics.precision_recall_curve(ood_indicator, -conf)

    precision_out, recall_out, thresholds_out \
        = metrics.precision_recall_curve(1 - ood_indicator, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out, fpr

def _save_scores( pred, conf, gt, save_name):
        save_dir = os.path.join('./results/', 'scores')
        os.makedirs(save_dir, exist_ok=True)
        np.savez(os.path.join(save_dir, save_name),
                 pred=pred,
                 conf=conf,
                 label=gt)
def _save_csv(metrics, dataset_name):
        [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
        best=1
        write_content = {
            'dataset': dataset_name,
            'FPR@95': '{:.2f}'.format(100 * fpr),
            'AUROC': '{:.2f}'.format(100 * auroc),
            'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
            'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
            'ACC': '{:.2f}'.format(100 * accuracy)
        }

        fieldnames = list(write_content.keys())

        # print ood metric results
        print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
              end=' ',
              flush=True)
        print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
            100 * aupr_in, 100 * aupr_out),
              flush=True)
        print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
        print(u'\u2500' * 70, flush=True)

        csv_path = os.path.join('./outputs_results/', 'ood_val.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(write_content)
        else:
            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow(write_content)
        
        if auroc>best:
            best=auroc
            save_point = './checkpoint/%s.pth.tar'%(args.id)
            save_checkpoint({
                'net':  net.state_dict(),
                'acc': best,
            }, save_point)  
############################################################################################
def val(epoch,iteration,train_loader: Dict[str,DataLoader],
                     val_loader: Dict[str, DataLoader],):
    if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
    else:
            net.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        print('#######################')
        total_batches = len(val_loader)
        print("Total batches in validation set:", total_batches)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        # outputs = net(inputs)

        id_pred, id_conf, id_gt = inference(net=net, data_loader=train_loader)
        _save_scores(id_pred, id_conf, id_gt, 'cifar 10')
        ood_pred, ood_conf, ood_gt = inference_val(net=net, data_loader=val_loader)
        _save_scores(ood_pred, ood_conf, ood_gt, 'cifar 10')
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics = compute_all_metrics(conf, label, pred)
        metrics_list = []
        _save_csv(metrics=ood_metrics, dataset_name='cifar 10')
        metrics_list.append(ood_metrics)
        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        _save_csv(metrics=metrics_mean, dataset_name='ood_val')
        
        # if acc > best:
        #     best = acc
        #     print('| Saving Best Model (net)...')
        #     save_point = './checkpoint/%s.pth.tar'%(args.id)
        #     save_checkpoint({
        #         'state_dict': net.state_dict(),
        #         'best_acc': best,
        #     }, save_point)

def val_tch(epoch,iteration):
    # global best
    # tch_net.eval()
    # val_loss = 0
    # correct = 0
    # total = 0
    # for batch_idx, (inputs, targets) in enumerate(val_loader):
    #     if use_cuda:
    #         inputs, targets = inputs.cuda(), targets.cuda()
    #     inputs, targets = Variable(inputs, volatile=True), Variable(targets)
    #     outputs = tch_net(inputs)
    #     loss = criterion(outputs, targets)

    #     val_loss += loss.item()
    #     _, predicted = torch.max(outputs.data, 1)
    #     total += targets.size(0)
    #     correct += predicted.eq(targets.data).cpu().sum()

    # # Save checkpoint when best model
    # acc = 100.*correct/total
    # print("| tch Validation Epoch #%d Batch #%3d\t\t\tLoss: %.4f Acc@1: %.2f%%\n" %(epoch, iteration, loss.item(), acc))
    # record.write(' | tchAcc: %.2f\n' %acc)
    # record.flush()
    # if acc > best:
    #     best = acc
    #     print('| Saving Best Model (tchnet)...')
    #     save_point = './checkpoint/%s.pth.tar'%(args.id)
    #     save_checkpoint({
    #         'state_dict': tch_net.state_dict(),
    #         'best_acc': best,
    #     }, save_point)        
    tch_net.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        # outputs = net(inputs)

        id_pred, id_conf, id_gt = inference(net=net, data_loader=train_loader)
        ood_pred, ood_conf, ood_gt = inference_val(net=net, data_loader=val_loader)
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])
        ood_metrics = compute_all_metrics(conf, label, pred)
        metrics_list = []
        _save_csv(metrics=ood_metrics, dataset_name='cifar 10')
        metrics_list.append(ood_metrics)
        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        _save_csv(metrics_mean, dataset_name='ood_val_tch')

########################################################################
from openood.evaluators.ood_evaluator import OODEvaluator
from openood.evaluators.base_evaluator import BaseEvaluator
from openood.postprocessors import BasePostprocessor
########################################################################
oodevaluator= OODEvaluator()

def test_id():
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(test_cifar10_loader, desc="Testing")):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = test_net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    test_id_acc = 100. * correct / total
    print('* Test id results : Acc@1 = %.2f%%' % (test_id_acc))
    record.write('\nTest id Acc: %f\n' % test_id_acc)
    record.flush()
    
def test_cifar100_ood():
    if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
    else:
            net.eval()
    for batch_idx, (inputs, targets) in enumerate(test_cifar100_loader):
        print('#######################')
        total_batches = len(test_cifar100_loader)
        print("Total batches in test set:", total_batches)
        print('batch index:', batch_idx)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        # outputs = net(inputs)

        id_pred, id_conf, id_gt = inference(net=net, data_loader=train_loader)
        _save_scores(id_pred, id_conf, id_gt, 'cifar10')
        ood_pred, ood_conf, ood_gt = inference_val(net=net, data_loader=test_cifar100_loader)
        _save_scores(ood_pred, ood_conf, ood_gt, 'cifar100')
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])
        # for sample in train_loader:
        #     data = sample['data'].cuda()
        #     label = sample['label'].cuda()
        # she_postprocessor=SHEPostprocessor()
        # she_postprocessor.setup(net, train_loader, test_cifar100_loader)
        # with torch.no_grad():
        #         id_pred, id_conf = she_postprocessor.postprocess(net, data)

        ood_metrics = compute_all_metrics(conf, label, pred)
        metrics_list = []
        _save_csv(metrics=ood_metrics, dataset_name='cifar 100')
        metrics_list.append(ood_metrics)
        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        _save_csv(metrics=metrics_mean, dataset_name='ood_cifar100_test')

def test_tin_ood():
    if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
    else:
            net.eval()
    for batch_idx, (inputs, targets) in enumerate(test_tin_loader):
        print('#######################')
        total_batches = len(test_tin_loader)
        print("Total batches in test set:", total_batches)
        print('batch index:', batch_idx)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        # outputs = net(inputs)

        id_pred, id_conf, id_gt = inference(net=net, data_loader=train_loader)
        _save_scores(id_pred, id_conf, id_gt, 'cifar10')
        ood_pred, ood_conf, ood_gt = inference_val(net=net, data_loader=test_tin_loader)
        _save_scores(ood_pred, ood_conf, ood_gt, 'tin')
        ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
        pred = np.concatenate([id_pred, ood_pred])
        conf = np.concatenate([id_conf, ood_conf])
        label = np.concatenate([id_gt, ood_gt])
        # for sample in train_loader:
        #     data = sample['data'].cuda()
        #     label = sample['label'].cuda()
        # she_postprocessor=SHEPostprocessor()
        # she_postprocessor.setup(net, train_loader, test_cifar100_loader)
        # with torch.no_grad():
        #         id_pred, id_conf = she_postprocessor.postprocess(net, data)

        ood_metrics = compute_all_metrics(conf, label, pred)
        metrics_list = []
        _save_csv(metrics=ood_metrics, dataset_name='tin')
        metrics_list.append(ood_metrics)
        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        _save_csv(metrics=metrics_mean, dataset_name='ood_tin_test')

record=open('./checkpoint/'+args.id+'.txt','w')
record.write('learning rate: %f\n'%args.lr)
record.write('batch size: %f\n'%args.batch_size)
record.write('start iter: %d\n'%args.start_iter)  
record.write('mid iter: %d\n'%args.mid_iter)  
record.flush()

###################################################
from read import train_CustomDataset 
from read import val_CustomDataset 
from read import test_cifar10_Dataset
from read import test_cifar100_Dataset 
from read import test_tin_Dataset 
from torch.utils.data import Dataset, DataLoader
################################################################################################
#Train#################

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

train_file_path = './data/benchmark_imglist/cifar10/train_cifar10.txt'
train_dataset = train_CustomDataset(train_file_path, transform = train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=1,drop_last=True)
#Val###############
val_transform=transforms.Compose([
    transforms.Resize(32,interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

val_file_path = './data/benchmark_imglist/cifar10/val_tin.txt'
val_dataset = val_CustomDataset(val_file_path, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=0)

#Test############################
#  test cifar 10 ##############################################
cifar10_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]),
])
cifar10_file_path = './data/benchmark_imglist/cifar10/test_cifar10.txt'
test_cifar10_dataset = test_cifar10_Dataset(cifar10_file_path,cifar10_transform)
test_cifar10_loader = DataLoader(test_cifar10_dataset, batch_size=10, shuffle=False, num_workers=1)
###############################################################
# test cifar 100###############################################
cifar100_transform =  transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
cifar100_file_path = './data/benchmark_imglist/cifar10/test_cifar100.txt'
test_cifar100_dataset = test_cifar100_Dataset(cifar100_file_path,cifar100_transform)
test_cifar100_loader = DataLoader(test_cifar100_dataset, batch_size=180, shuffle=False, num_workers=1)
###############################################################
# test tin ####################################################
tin_transform = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
tin_file_path = './data/benchmark_imglist/cifar10/test_tin.txt'
test_tin_dataset = test_tin_Dataset(tin_file_path,tin_transform)
test_tin_loader = DataLoader(test_tin_dataset, batch_size=200, shuffle=False, num_workers=1)
###############################################################
###################################################################################################################
best = 0
init = True
# Model
print('\nModel setup')
print('| Building net')

# from torch.hub import load_state_dict_from_url
# model_urls = {
#     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
#     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
#     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
#     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
#     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
# }
# pretrained_dict = load_state_dict_from_url(model_urls['resnet50'])

# # 移除全连接层的权重和偏置
# pretrained_dict.pop('fc.weight', None)
# pretrained_dict.pop('fc.bias', None)

net = models.resnet50()
# net.linear = nn.Linear(2048,10)
tch_net = models.resnet50()
# tch_net.fc = nn.Linear(2048,10)
pretrain_net = models.resnet50()
# pretrain_net.fc = nn.Linear(2048,10)
test_net = models.resnet50()
# test_net.fc = nn.Linear(2048,10)

print('| load pretrain from checkpoint...')
checkpoint_path = './checkpoint/%s.pth.tar' % args.checkpoint
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['net']

num_features = 2048 
num_classes = 10     

params = net.state_dict()
weight_dic = {}

# for k,v in params.items():
#     print(k)
# print('******************')
model_state_dict = checkpoint['net']
# for key in model_state_dict.keys():
#     print(key)
net.load_state_dict(state_dict)

if use_cuda:
    net.cuda()
    tch_net.cuda()
    pretrain_net.cuda()
    test_net.cuda()
    cudnn.benchmark = True
# pretrain_net.eval()    

for param in tch_net.parameters(): 
    param.requires_grad = False   
for param in pretrain_net.parameters(): 
    param.requires_grad = False 

criterion = nn.CrossEntropyLoss()
consistent_criterion = nn.KLDivLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3)

print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))

for epoch in range(1, 1+args.num_epochs):
    train(epoch)

    print('\nTesting model')
    checkpoint_path = './checkpoint/%s.pth.tar' % args.checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # if 'state_dict' in checkpoint:
    #     checkpoint['state_dict'].pop('state_dict.weight')
    #     checkpoint['state_dict'].pop('state_dict.bias')
    test_net.load_state_dict(state_dict)

    test_id()
    test_cifar100_ood()
    test_tin_ood()

record.close()
