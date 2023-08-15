# <center>OPENCV学习笔记
## 图像
### 坐标系
坐标原点在左上角

每一个点的坐标是（height,width,channel） 

一张彩色的图片的channel是3：每张图每个像素有3个颜色组合而成（黑白图片的channel为2）
### opencv基本操作
读取图片:`cv2.imread`

获取图片的形状：`img.shape`，返回一个(roew,heights,channels)

获取图片的大小：`img.size`, 返回一个rows*heights*channels

显示图片：`cv2.imshow()`

等待：`cv2.waitKey(0)`

关闭：`cv2.destroyAllWindows()`

#### 彩色图片：
```python
import cv2#导入库
img = cv2.imread('./风景.jpg')
img.shape#读取图片形状
img_dtype=img.dtype#返回数据类型
#（B，G，R）
(b,g,r)=img[2,6]
#取色
b=img[2,6,0]
g=ing[2,6,1]
r=img[2,6,2]
#重新给像素赋值，更换颜色
img[2,6]=(0,,0,255)
#显示图片
cv2.imshow("image",img)
#等待一定时间
cv2.waitKey(0)
#关闭窗口
cv2.destroyAllWindows()
```
 灰度图片的通道默认是0

#### BGR顺序
```python
import cv2
import matplotlib as plt
img_logo=cv2.imread('./logo.jpg')
b,g,r=cv2.split(img_logo)
img_new=cv2.merge([r,g,b])#改变r,g,b顺序
plt.subplot(121)
plt.imshow(img_logo)#绘制图像
plt.subplot(122)
plt.imshow(img_new)
plt.show()#展示图像
```
### 读取、显示图片
```python
import cv2
import argparse

#获取参数
parser=argparse.ArgumentParser()

#添加参数
parser.add_argument("-path_image",help="path to input the image") #通过终端传参数

#解析参数
args=parser.parse_args()

#加载图片 方式一
img=cv2.imread(args.path_image)
cv2.imshow("logo",img)

#加载图片 方式二
args_dict=vars(parser.parse_args()) #以字典形式
img2=cv2.imread(args_dict["path_image"])
cv2.imshow("logo_two",img2)

#等待
cv2.waitKey(0)

#关闭
cv2.destroyWindow()
```
### 读取、处理、保存图片
```python
import cv2
import argparse

#获取参数
parser=argparse.ArgumentParser()

#添加参数
parser.add_argument("img_input",help="read one image")
parser.add_argument("img_output",help="save the processed image")

#解析参数，用字典形式保存参数和值
args=vars(parser.parse_args())

#加载图片
img=cv2.imread(args["img_input"])

#灰度处理
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# 保存图片
cv2.imwrite(args["img_output"],img_gray)

#显示图片
cv2.imshow("Original picture",img)
cv2.imshow("Gray picture",img_gray)

#等待
cv2.waitKey(0)

#关闭窗口
cv2.destroyWindow()
```

### 图像变换
```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

#图像的加载
img=cv2.imread("picture.jpg")
plt.imshow(img)

#获取图片高，宽，颜色通道
height,width,channel=img.shape
print(height,width,channel)

#图片的放大缩小
resized_img=cv2.resize(img,(width*2,height*2),interpolation=cv2.INTER_LINEAR)#双线性插值算法
plt.imshow(resized_img)

small_img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)#长宽缩小0.5倍，双线性插值算法
plt.imshow(small_img)

#图像的平移
height,width=img.shape[:2]
M1=np.float32([[1,0,200],[0,1,50]])#平移矩阵，向右移200像素，向下移动50像素
move_img=cv2.warpAffine(img,M1,(width,height))
plt.imshow(move_img)

#图像的旋转
height,width=img.shape[:2]
center=(width//2.0,height//2.0)
M2=cv2.getRotationMatrix2D(center,180,1)#1:旋转过程中无缩放
rotation_img=cv2.warpAffine(img,M2,(width,height ))
plt.imshow(rotation_img)


#图像的仿射变换
p1=np.float32([[120,35],[215,45],[135,120]])
p2=np.float32([[135,45],[300,110],[130,230]])
M3=cv2.getAffineTransform(p1,p2)#计算一个变换矩阵
trans_img=cv2.warpAffine(img,M3,(width,height))
plt.imshow(trans_img)

#图形的裁剪
crop_img=img[20:60,200:400]
plt.imshow(crop_img)

#位运算
#长方形
rectangle=np.zeros((300,300),dtype='uint8')#创建画框
rect_img=cv2.rectangle(rectangle,(25,25),(275,275),255,-1)
plt.imshow(rect_img)

#圆形
circle=np.zeros((300,300),dtype='uint8')
circle_img=cv2.circle(circle,(150,150),150,255,-1)
plt.imshow(circle_img)

and_img=cv2.bitwise_and(rect_img,circle_img)
plt.imshow(and_img)

#或运算 0，1：1， 1，0：1， 0，0：0， 1，1：1
or_img=cv2.bitwise_or(rect_img,circle_img)
plt.imshow(or_img)

#异或运算 01：1 10：1 00：0 11：0
xor_img=cv2.bitwise_xor(rect_img,circle_img)
plt.imshow(xor_img)

#图像的分离和融合
#分离
(B,G,R)=cv2.split(img)
plt.imshow(B)

#融合
zeros=np.zeros(img.shape[:2],dtype='uint8')
plt.imshow(cv2.merge([zeros,zeros,R]))

#图像的颜色空间
gray=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
plt.imshow(gray)

#hsv
hsv=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.imshow(hsv)

#lab
lab=cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
plt.imshow(lab)
plt.show()
```
### 灰度直方图
##### 定义、意义、特征
定义：二位统计图表
意义：像素分布强度的图形表达方式，统计了每一个强度值所具有的像素个数

```python
import cv2
import matplotlib.pyplot as plt


def show_image(image,title,position):
    #BGR 到 RGB
    img_RGB=image[:,:,::-1]
    #显示标题
    plt.title(title)
    plt.subplot(2,2,position)#定位显示
    plt.imshow(img_RGB)
    plt.show()


#显示灰度直方图
def show_histogram(hist,title,postition,color):
    #显示标题
    plt.title(title)
    plt.subplot(2,3,postition)
    plt.xlabel("Bins")#横轴名称
    plt.ylabel("Pixels")#纵轴名称
    plt.xlim([0,256])#范围
    plt.plot(hist,color=color)



#创建画布
plt.figure(figsize=(15,6))#画布大小
plt.suptitle("灰度直方图")#图像名称

#加载图片
img=cv2.imread("picture.jpg")

#灰度转换
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#计算灰度图的直方图
hist_img=cv2.calcHist([img_gray],[0],None,[256],[0,256])

img_BGR=cv2.cvtColor(img_gray,cv2.COLOR_GRAY2BGR)
show_image(img_BGR,"RGB image",1)
show_histogram(hist_img,"gray image histogram",4,'m')

plt.show()
```

### 彩色直方图
```python
import cv2
import matplotlib.pyplot as plt

def show_image(image,title,position):
    #BGR 到 RGB
    img_RGB=image[:,:,::-1]
    #显示标题
    plt.title(title)
    plt.subplot(2,2,position)#定位显示
    plt.imshow(img_RGB)


#显示彩色直方图
def show_histogram(hist,title,postition,color):
    #显示标题
    plt.title(title)
    plt.subplot(2,3,postition)
    plt.xlabel("Bins")#横轴名称
    plt.ylabel("Pixels")#纵轴名称
    plt.xlim([0,256])#范围
    for h,c in zip(hist,color):
        plt.plot(h,color=c)

def cal_color_hist(image):
    hist=[]
    hist.append(cv2.calcHist([image],[0],[256],[0,256]))
    hist.append(cv2.calcHist([image],[1],[256],[0,256]))
    hist.append(cv2.calcHist([image], [2], [256], [0, 256]))
    return hist

#创建画布
plt.figure(figsize=(12,8))#画布大小
plt.suptitle("Color Histogram")#图像名称

#加载图片
img=cv2.imread("picture.jpg")

#计算直方图
img_hist=cal_color_hist(img)

#显示图片
show_image(img,"RGB image",1)
show_histogram(img_hist,"color image histogram",2,('b','g','r'))

plt.show()
```