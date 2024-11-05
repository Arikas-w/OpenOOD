from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import models_test as models

import os
import sys
import time
import argparse
import datetime

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch Clothing-1M Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning_rate')
parser.add_argument('--start_epoch', default=2, type=int)
parser.add_argument('--num_epochs', default=1, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--optim_type', default='SGD')
parser.add_argument('--seed', default=7)
parser.add_argument('--gpuid', default=1, type=int)
parser.add_argument('--id', default='cross_entropy')
args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
  
# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    learning_rate = args.lr
    if epoch > args.start_epoch:
        learning_rate=learning_rate/10        
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

    print('\n=> %s Training Epoch #%d, LR=%.4f' %(args.id,epoch, learning_rate))
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs =net(inputs)              # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, args.num_epochs, batch_idx+1, (len(train_loader.dataset)//args.batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()
        # if (batch_idx+1)%1000==0:
        #     val(epoch)
        #     net.train()
            
def val(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100.*correct/total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    record.write('Validation Acc: %f\n'%acc)
    record.flush()
    if acc > best_acc:
        best_acc = acc
        print('| Saving Best Model ...')
        save_point = './checkpoint/%s.pth.tar'%(args.id)
        save_checkpoint({'state_dict': net.state_dict(),}, save_point) 


def test():
    global test_acc
    test_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = test_net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    acc = 100.*correct/total   
    test_acc = acc
    record.write('Test Acc: %f\n'%acc)
    
os.mkdir('checkpoint')     
record=open('./checkpoint/'+args.id+'_test.txt','w')
record.write('learning rate: %f\n'%args.lr)
record.flush()


from read_baseline import train_CustomDataset 
from read_baseline import val_CustomDataset 
from read_baseline import test_CustomDataset
########################################################
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
train_file_path = './data/benchmark_imglist/cifar10/train_cifar10.txt'
train_dataset = train_CustomDataset(train_file_path, transform = train_transform)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
val_file_path = './data/benchmark_imglist/cifar10/val_cifar10.txt'
val_dataset = val_CustomDataset(val_file_path, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
test_file_path = './data/benchmark_imglist/cifar10/test_cifar10.txt'
test_cifar10_dataset = test_CustomDataset(test_file_path,test_transform)
test_loader = DataLoader(test_cifar10_dataset, batch_size=64, shuffle=False, num_workers=4)
########################################################

best_acc = 0
test_acc = 0
# Model
print('\nModel setup')
print('| Building net')
net = models.resnet50(pretrained=False)
###########################################################
from torch.hub import load_state_dict_from_url
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
pretrained_dict = load_state_dict_from_url(model_urls['resnet50'])

# 移除全连接层的权重和偏置
pretrained_dict.pop('fc.weight', None)
pretrained_dict.pop('fc.bias', None)

# 将预训练的 state_dict 应用到模型（除了全连接层）
net.load_state_dict(pretrained_dict, strict=False)

# # 重新初始化全连接层的权重和偏置
net.fc.weight.data = torch.nn.init.xavier_uniform_(net.fc.weight.data)
net.fc.bias.data = torch.nn.init.zeros_(net.fc.bias.data)
##################################################################
net.state_dict = nn.Linear(2048,10)
test_net = models.resnet50(pretrained=False)
test_net.load_state_dict(pretrained_dict, strict=False)
test_net.fc = nn.Linear(2048,10)
if use_cuda:
    net.cuda()
    test_net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


print('\nTraining model')
print('| Training Epochs = ' + str(args.num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(args.optim_type))

for epoch in range(1, 1+args.num_epochs):
    train(epoch)
    val(epoch)
    scheduler.step()

print('\nTesting model')
checkpoint_path = './checkpoint/'
file_name = 'cross_entropy.pth.tar'
file_path = os.path.join(checkpoint_path, file_name)
os.makedirs(checkpoint_path, exist_ok=True)

checkpoint = torch.load('./checkpoint/%s.pth.tar'%args.id)
#################
num_features = 2048 
num_classes = 10     

state_dict = checkpoint['state_dict']

if 'state_dict' in checkpoint:
    checkpoint['state_dict'].pop('state_dict.weight')
    checkpoint['state_dict'].pop('state_dict.bias')

test_net.load_state_dict(state_dict)

test()

print('* Test results : Acc@1 = %.2f%%' %(test_acc))
record.write('Test Acc: %.2f\n' %test_acc)
record.flush()
record.close()