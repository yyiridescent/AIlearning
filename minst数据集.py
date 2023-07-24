from torch.utils.data import Dataset,Dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
import numpy as np
import cv2
import torch.nn.functional as F



#参数
BATCH_SIZE=16#处理的数据量
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS=10#训练的轮次

# class MyData(Dataset):
#
#     # 获取图片地址列表
#     def __init__(self,root_dir,label_dir):#初始化类，提供全局变量
#         self.root_dir=root_dir
#         self.label_dir=label_dir
#         self.path=os.path.join(self.root_dir,self.label_dir)#join 连接两个地址,要从pythonProject的根目录开始算起
#         self.img_path=os.listdir(self.path)#获得图片所有的地址
#
#
#     #确定每个图片的位置
#     def __getitem__(self, idx):#获取其中每个图片的地址   idx序号
#         img_name=self.img_path[idx]#定位图片在文件夹中的位置
#         img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)#定位图片在pythonProject中的位置
#         img=Image.open(img_item_path)
#         label=self.label_dir
#         return img,label
#
#     #所有图片的数量
#     def __len__(self):
#         return len(self.img_path)


#构建pipeline
pipeline=transforms.Compose({
    transforms.Totensor(),
    transforms.Normalize((0.1307,),(0.3081,))
})

#下载数据集
train_set=datasets.MNIST("minst数据集",train=True,download=True,transform=pipeline)
test_set=datasets.MNIST("minst数据集",train=False,download=True,transform=pipeline)

# 数据加载
train_loader=Dataloader(train_set,batch_size=BATCH_SIZE,shuffle=True)#图片顺序打乱
test_loader=Dataloader(test_set,atch_size=BATCH_SIZE,shuffle=True)

#转化成jpg格式
with open ("./minst数据集/train-images-idx3-ubyte","rb") as f:
    file=f.read()

image1=[int(str(item).encode('ascil'),16) for item in file[16:16+784]]
# print(image1)

image1_np=np.array(image1,dtype=np.uint8).reshape(28,28,1)
# print(image1_np.shape)
cv2.imwrite("digit.jpg",image1_np)



# 模型的构建
class Digit(nn.Moudle):
    # 构造方法
    def __init__(self):
        super().__init__()
        # 属性
        self.conv1 = nn.Conv2d(1, 10, 5)  # 1：灰度图片的通道 10：输出通道 5：卷积核
        self.conv2 = nn.Conv2d(10, 20, 3)  # 10：输入通道 20：输出通道 3：卷积核
        self.linear1 = nn.Linear(20 * 10 * 10, 500)  # 全连接层   输入通道；输出通道
        self.linear2 = nn.Linear(500, 10)  # 输入通道 输出通道

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(x)  # 激活函数 打破线性
        x = F.max_pool2d(x, 2, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(input_size, -1)

        x = self.linear1(x)
        x = F.relu(x)  # 保持shape
        x = self.linear2(x)
        output = F.log_softmax(x, dim=1)  # 计算分类后的概率值
        return output



#优化器
model=Digit().to(DEVICE)

optimizer=optim.Adam(model.parameters())#adam优化器


def train_model(model,device,train_loader,optimizer,epoch):
    #模型训练
    model.train()
    for batch_index,(data,label) in enumerate(train_loader):
        data,label=data.to(device),label.to(device)#部署device
        #梯度初始为0
        optimizer.zero_grad()
        #训练后结果（预测）
        output=model(data)
        #损失
        loss=F.cross_entropy(output,label)
        pred=output.max(1,keepim=True)
        loss.backword() #反向
        optimizer.step()#参数优化



def test_model(model,device,test_loader):
    model.eval()#验证
    with torch.no_grad():
        for data,label in test_loader:
            data,label=data.to(device),label.to(device)
            output=model(data)
            # 损失
            loss = F.cross_entropy(output, label).item()
            #概率最大的值的下标
            pred=output.amx(1,keepdim=True)[1]
            #累计正确率
            correct=pred.eq(label.view_as(pred)).sum().item()
        loss=loss/len(test_loader.dataset)
        print(loss)
        print(correct)



for epoch in range(1,EPOCHS+1):
    train_model(model,DEVICE,train_loader,optimizer,epoch)
    test_model(model,DEVICE,test_loader)