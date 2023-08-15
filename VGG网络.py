import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 降低过拟合
            nn.Linear(512 * 7 * 7, 2048),  # 第一层全连接层
            nn.ReLU(True),  # 激活函数
            nn.Dropout(p=0.5),  # 两个全连接层的连接
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, class_num)  # 类别个数
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weight(self):#初始化权重函数
        for m in self.Module():  # 遍历子模块
            if isinstance(m, nn.Conv2d):  # 卷积层
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(nn.bias, 0)
            elif isinstance(m, nn.Linear):  # 全连接层
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x=self.features(x)
        x=torch.flatten(x,start_dim=1)#展平，从1的维度开始展品
        x=self.classifier(x)
        return x


# 提取特征网络结构
def make_features(cfg: list):
    layers = []  # 存放创建的每一层结构
    in_channels = 3
    for v in cfg:
        if v == 'M':  # 最大池化层
            layers = layers + [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers = layers + [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)  # 非关键字参数


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [4, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

#实例化给定的配置模型
def vgg11(model_name="vgg11",**kwargs):
    try:
        cfg=cfgs[model_name]
    except:
        exit(-1)
    model=VGG(make_features(cfg),**kwargs)
    return model

def vgg13(model_name="vgg13",**kwargs):
    try:
        cfg=cfgs[model_name]
    except:
        exit(-1)
    model=VGG(make_features(cfg),**kwargs)
    return model

def vgg16(model_name="vgg16",**kwargs):
    try:
        cfg=cfgs[model_name]
    except:
        exit(-1)
    model=VGG(make_features(cfg),**kwargs)
    return model

def vgg19(model_name="vgg19",**kwargs):
    try:
        cfg=cfgs[model_name]
    except:
        exit(-1)
    model=VGG(make_features(cfg),**kwargs)
    return model

def test():
    net=vgg13()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x=torch.randn(2,3,224,224)
    y=net(x).to(device)
    print(y.shape)

test()