import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as mp
import time

start=time.time()

class block(nn.Module):
    def __init__(self,in_channel,out_channel,identity_downsample=None,stride=1):
        super(block,self).__init__()
        self.expension=4
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False,padding=0)#改变了输出通道数
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)#改变特征图大小
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * 4,
                               kernel_size=1, stride=1, bias=False,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.identity_downsample = identity_downsample#改变identity的通道数，使之与out的通道数相匹配


    def forward(self, out):
        identity = out

        if self.identity_downsample is not None:
            identity = self.identity_downsample(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        # print(out.shape)

        return out


class ResNet(nn.Module):
    def __init__(self,block,layers,img_channels,num_classes):
        super(ResNet,self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(img_channels, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block,layers[0], out_channels=64,stride=1)
        self.layer2 = self._make_layer(block,layers[1], out_channels=128,stride=2)
        self.layer3 = self._make_layer(block,layers[2], out_channels=256,stride=2)
        self.layer4 = self._make_layer(block,layers[3], out_channels=512,stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*4,num_classes)

    def forward(self, out):
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out=self.layer1(out)
        out=self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out=self.avgpool(out)
        out=out.reshape(out.shape[0],-1)
        out=self.fc(out)
        # print(out.shape)
        return out


    def _make_layer(self,block,num_residual_blocks,out_channels,stride):
        identity_downsample=None
        layers=[]

        if stride !=1 or self.in_channel !=out_channels*4:
            identity_downsample=nn.Sequential(nn.Conv2d(self.in_channel,out_channels*4,kernel_size=1,stride=stride),nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channel,out_channels,identity_downsample,stride))
        self.in_channel=out_channels*4

        for i in range(num_residual_blocks-1):
            layers.append(block(self.in_channel,out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,6,3],img_channels,num_classes)

def ResNet101(img_channels=3,num_classes=1000):
    return ResNet(block,[3,4,23,3],img_channels,num_classes)

def ResNet152(img_channels=3,num_classes=1000):
    return ResNet(block,[3,8,36,3],img_channels,num_classes)

#测试
def test():
    net=ResNet101()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x=torch.randn(2,3,224,224)
    y=net(x).to(device)
    print(y.shape)
    end=time.time()
    total=end-start
    print(total)

test()

# 作loss和accuracy的图
def picture(x_vals_1,y_vals_1,x_vals_2,y_vals_2,x_label,y_label1,y_label2,title,figsize=(3.5,2.5)):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体
    plt.rcParams['axes.unicode_minus'] = False  # 设置正负号

    # 设置画板
    fig = plt.figure(figsize=figsize, dpi=80)
    # 添加Axes坐标轴实例，创建1个画板
    ax = fig.add_subplot(111)
    # 制作第一条折现
    lin1 = ax.plot(x_vals_1, y_vals_1, label=x_label, color='r')
    ax.set_xlabel(x_label)
    # 设置Y轴1
    ax.set_ylabel(y_label1)
    # 使用twinx()函数实现共用一个x轴
    ax2 = ax.twinx()
    # 制作第二条折现
    lin2 = ax2.plot(x_vals_2, y_vals_2, label=y_label2, color='blue')
    # 设置Y轴2
    ax2.set_ylabel(y_label2)
    # 合并图例
    lines = lin1 + lin2
    labs = [label.get_label() for label in lines]
    ax.legend(lines, labs)
    # 增加网格线
    ax.grid()
    #图表标题
    plt.title(title)

    plt.show()