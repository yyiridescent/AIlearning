import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class DensenetLayer(nn.Sequential):
    def __init__(self,num_input_features,growth_rate,bn_size,drop_rate):
        super(DensenetLayer,self).__init__()
        self.add_module("noral",nn.BatchNorm2d(num_input_features))
        self.add_module("relu_1",nn.ReLU(inplace=True))
        self.add_module("conv1",nn.Conv2d(num_input_features,bn_size*growth_rate,kernel_size=1,stride=1,bias=False))#调整channels
        self.add_module("norm",nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu_2",nn.ReLU(inplace=True))
        self.add_module("cnv2",nn.Conv2d(bn_size*growth_rate,growth_rate,kernel_size=3,stride=1,padding=1,bias=False))
        self.drop_rate=drop_rate

    def forward(self,x):
        new_features=super(DensenetLayer,self).forward(x)
        if self.drop_rate>0:
            new_features=F.dropout(new_features,p=self.drop_rate,training=self.training)
        return torch.cat([x,new_features],1)

#循环
class Block(nn.Sequential):
    def __init__(self,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
        super(Block,self).__init__()
        for i in range(num_layers):
            layer=DensenetLayer(num_input_features+i*growth_rate,growth_rate,bn_size,drop_rate)
            self.add_module("denselayer%d"%(i+1),layer)


class Transition(nn.Sequential):
    def __init__(self,num_input_features,num_output_features):
        super(Transition,self).__init__()
        self.add_module("noral", nn.BatchNorm2d(num_output_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv",
                        nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module("pool", nn.AvgPool2d(kernel_size=2,stride=2))#降低size


class Densenet(nn.Module):
    def __init__(self,growth_rate=32,block_config=(6,12,24,16),num_init_features=64,bn_size=64,drop_rate=0,num_class=1000):
        super(Densenet,self).__init__()

        self.features=nn.Sequential(OrderedDict([
            ('conv0',nn.Conv2d(3,num_init_features,kernel_size=7,stride=2,padding=3,bias=False)),
            ('norm0',nn.BatchNorm2d(num_init_features)),
            ("relus",nn.ReLU(inplace=True)),
            ('pool0',nn.MaxPool2d(kernel_size=3,stride=2,padding=1))
        ]))

        num_features=num_init_features
        for i,num_layers in enumerate(block_config):
            block=Block(num_layers=num_layers,num_input_features=num_features,bn_size=bn_size,growth_rate=growth_rate,drop_rate=drop_rate)
            self.features.add_module("denseblock%d"%(i+1),block)
            num_features=num_features+num_layers*growth_rate
            if i !=len(block_config)-1:
                trans=Transition(num_input_features=num_features,num_output_features=num_features//2)
                self.features.add_module("transition%d"%(i+1),trans)
                num_features=num_features//2

        self.features.add_module("norm5",nn.BatchNorm2d(num_features))

        self.classifier=nn.Linear(num_features,num_class)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
            elif isinstance(m,nn.Linear):
                m.bias.data.zero_()

    def forward(self,x):
        features=self.features(x)
        out=F.relu(features,inplace=True)
        out=F.avg_pool2d(out,kernel_size=7,stride=1).view(features.size(0),-1)
        out=self.classifier(out)
        return out


def densenet121(**kwargs):
    model=Densenet(num_init_features=64,bn_size=32,block_config=(6,12,24,16),**kwargs)
    return model

def densenet169(**kwargs):
    model=Densenet(num_init_features=64,bn_size=32,block_config=(6,12,32,32),**kwargs)
    return model

def test():
    net=densenet121()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x=torch.randn(2,3,224,224)
    y=net(x).to(device)
    print(y.shape)


test()