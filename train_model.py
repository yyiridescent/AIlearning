import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision
from torch.utils.data import DataLoader
from ResNet import ResNet101,picture
import torch.nn as nn
import torch.optim as optim
from VGG网络 import vgg13
import time
from densenet_model import densenet121

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#计时
start=time.time()

#定义超参数
epoch=3
batch_size=64

#数据集的加载
train_transform=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
test_transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = torchvision.datasets.CIFAR100(root='D:\\PycharmProjects\\pythonProject1\\CIFAR\\cifar\\train', train=True,download=True, transform=train_transform)
test_data= torchvision.datasets.CIFAR100(root='D:\\PycharmProjects\\pythonProject1\\CIFAR\\cifar\\test',download=True,transform=test_transform)

#数据加载
train_loader=DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True,num_workers=0)
test_loader=DataLoader(dataset=test_data,batch_size=batch_size,shuffle=True,num_workers=0)


#模型
model=ResNet101()

#损失函数
criterion=nn.CrossEntropyLoss()

#优化器
optimizer=optim.Adam(model.parameters(),lr=0.001)

#cpu or gpu
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model=model.to(device)

#模型训练
train_loss=[]
for epoch in range(epoch):
    for i,data in enumerate(train_loader):
        #取数据和标签
        inputs,labels=data
        #数据和标签进入cpu/gpu
        inputs,labels=inputs.to(device),labels.to(device)
        #正向传播
        ouput=model(inputs)
        #损失函数
        loss=criterion(ouput,labels)
        #优化器，清空上一轮的梯度
        optimizer.zero_grad()

        #反向传播
        loss.backward()

        optimizer.step()

        loss = loss.item()
        train_loss.append(loss)

        if i % 100 == 0:
            print('loss:{:.4f}'.format(loss))

model.train()

#模型预测
correct,total=0.0,0.0
train_accuracy=[]
for j,data in enumerate(test_loader):
    # 取数据和标签
    inputs, labels = data
    # 数据和标签进入cpu/gpu
    inputs, labels = inputs.to(device), labels.to(device)
    # 正向传播
    ouput = model(inputs)
    _,predicts=torch.max(ouput.data,1)
    total=total+labels.size(0)
    correct = correct +(predicts == labels).sum().item()
    train_accuracy.append(correct)

end=time.time()
total_time=int(end-start)
len_train_loss=len(train_loss)
len_train_accuracy=len(train_accuracy)

picture(np.linspace(start=0,stop=total_time,num=len_train_loss,endpoint=True),train_loss,np.linspace(start=0,stop=total_time,num=len_train_accuracy,endpoint=True),train_accuracy,'time','train_loss','train_accuracy','loss and accuracy')
