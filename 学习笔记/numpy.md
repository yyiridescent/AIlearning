```python
import numpy as np

a1=np.array([1,2,3])#[1 2 3]
a2=np.array([[1,2],[3,4]])#[[1,2] / n [3 4]]
a3=np.array([1,2,3,4,5],ndmin=2)#[[1 2 3 4 5]]
a4=np.array([1,2,3],dtype=bool)#[True True True] complex同理

b1=np.array([[1,2,3],[4,5,6]])#2行3列
b2=b1.reshape(3,2)#变成3行2列
b3=b1.reshape(2,-2)#固定两行，自动计算列数 同理（负数，2）固定两列
b4=b1.reshape(1,2,3)#返回三维数组

c1=np.arange(5)#一个参数，最后一个数为5-1=4，间隔为1，返回数组，默认int型（下同理）
c2=np.arange(5,10)#两个参数，起始值为5，最后一个数为10-1=9，间隔为1，返回数组
c3=np.arange(1,10,2)#三个参数，起始值为1，间隔为2，最后一个数小于10
c4=c1.ndim#返回c1的维数=秩
c5=b1.shape#返回b1的型号（2，3）——2行3列
b1.shape=(3,2)#将b1的数组转换成3行2列

d1=np.zeros([3,2],dtype=int,order='C')#0数组，返回int类型（可选，默认为float），C为行优先，F为列优先
d2=np.ones([1,5],dtype=int)#返回全是1的数组

e1=[1,2,3,4,5]
e2=np.asarray(e1)#列表转化为ndarray
e3=(1,2,3,4,5)
e4=np.asarray(e3)#元组转化为ndarray
e5=b'hello  world'#使用frombuffer时要把字符串转化为bytestring类型，要在前面加b
e6=np.frombuffer(e5,dtype='S1')#S1每次输出一个，S2每次输出两个，若字符串为单数，会报错

f1=np.linspace(1,10,10)#创造等差数列
#起始值为1，末尾值为10（默认包含，若endpoint为false则不包含），其中有10个数（中间等长），dtype默认为float
f2=f1.reshape([10,1])#将f1转化为10行1列的数组
f3=np.logspace(1.0,2.0,20,base=2)
#默认底数为10，即10至100中取10个等比的数,底数为2（默认为10），dtype为数据类型

g1=np.arange(10)
g2=slice(2,7,2)#从2开始索引，最大小于7，间隔为2
g3=g1[g2]#输出[2 4 6]
g4=g1[2:7:2]#和g2 g3同理，输出[2 4 6]
g5=g1[2]#返回对应的单个元素
g6=g1[2:]#返回2之后的所有数
g7=g1[2:7]#返回两个数之间的数（不包括7）
g8=np.array([[1,2,3],[2,3,4],[3,4,5]])
g9=g8[1:]#返回g8[1：]之后的数组  [[2 3 4] /n [3 4 5]]
g10=g8[...,1]#返回第二列元素
g11=g8[1,...]#返回第二行元素
g12=g8[...,1:]#返回第二列及剩下的元素
g13=g8[[0,1,2],[0,1,0]]#返回（0，0）（1，1）（2，0）位置的元素
g14=g8[g8>3]#输出g8中大于3的元素
g15=np.array([np.nan,1,np.nan,9])
g16=g15[~np.isnan(g15)]#提出np.nan iscomplex 剔除复数
g17=np.arange(32).reshape(8,4)
g18=g17[[4,2,1,7]]#输出对应行,负数为倒数第几行
g19=g17[np.ix_([1,5,7,2],[0,3,1,2])]#返回4*4矩阵 第一行为g17[1,0] g17[1,3] g17[1,1] g17[1,2],以此类推

i1=np.array([1,2,3,4])
i2=np.array([10,20,30,40])
i3=i1*i2#返回[10 40 90 160] 若两个矩阵型号相同，则对应数据相乘
i4=np.array([[0,0,0],
             [10,10,10],
             [20,20,20],
             [30,30,30]])
i5=np.array([1,2,3])
i5=i4+i5#输出每一行都加上i1

j1=np.arange(6).reshape(2,3)
j2=j1.T#j1的转置
#for i in np.nditer(j1):#用nditer迭代
#for i in np.nditer(j1,order='F') 按列优先 默认按行优先
    #print(i,end=' ')#输出数组中的每个数字 以行优先
#若在遍历数组的时候，要修改数组元素  for i in np.nditer(j1,op_flags=['readwrite'])
#按列输出一维数组  for i in np.nditer(j1,flags=['external_loop'],order='F')
#另一种迭代器,输出每一个数据 for i in j1.flat
#展开为一维数组 for i j1.flatten(order='F') 按列展开 默认按行展开
j3=np.transpose(j1)#相当于转置

#k0=np.array([1],[2])
k1=np.array([[1,2],[3,4]])
k2=np.array([[11,12],[21,22]])
k3=np.dot(k1,k2) #两个矩阵的乘积
k4=np.vdot(k1,k2)#1*11+2*12+3*21+*22
k5=np.inner(k1,k2)#内积
k6=np.linalg.det(k1)#返回k1行列式的值
k7=np.linalg.inv(k1)#返回k1的逆矩阵
#k8=np.linalg.solve(k1,k0)#线性方程组的值系数矩阵为k1

l1=np.matlib.empty((2,2))#2*2的零矩阵  ones同理
l2=np.matlib.eye(n=3,M=4)#最后一列全为0
l3=np.matlib.identity(3)#返回3*3的单位矩阵 默认为float型

m0=1.8023
m_=np.array([2.4,3.4,2,7,9])
m1=np.around(m0,decimals=1)#舍入1位小数，默认为0
#python自带sin() cos() tan() arcsin() arccos() arctan()
m2=np.floor(m_)#输出的数组中为小于等于数组中每个数字的最大整数
m2=np.ceil(m_)#输出的数组中为大于等于数组中每个数字的最大整数

n0=np.array([1,2,3])
n1=np.array([[0,1,2],[3,4,5],[6,7,8]])
n2=np.array([10,10,10])
#numpy中有add() subtract() multiply() divide()
n3=np.reciprocal(n2)#求所有元素的倒数
n4=np.power(n0,n2)#以第一个数组为底数，第二个数组作为幂指数，返回计算结果
n5=np.mod(n2,n0)#n2除以n0中的对应的数字之后的余数  和np.remainder()

o1=np.array([[1,2,3],[2,3,4],[3,4,5]])
o2=np.amin(o1,1)#按行得出最小值
o3=np.amin(o1,0)#按列得出最小值 np.amax同理
o4=np.ptp(o1)#返回最大值与最小值的差，可研axis=1按行求极差
o5=np.percentile(o1,50)#o1的50%分位数 可用axis=0或1按列or行求出对应列or行的50%分位数
o6=np.mean(o1)#o1的平均值
o7=np.array([1,2,3,4])
o8=np.array([4,3,2,1])
o9=np.average(o7,weights=o8)#o7对于o8的加权平均值
o10=np.std([1,2,3,4])#求标准差
o11=np.var([1,2,3,4])#求方差

p0=np.array([3,2,1,4])
p1=np.array([[3,7],[1,9]])
p2=np.sort(p1)#按行排序 后添加axis=0则按列排序
p3=np.argsort(p0)#返回排序后的索引值

q1=(10,20,30,40)
q2=(20,50,40.30)
q3=(40,70,90,70)
ind=np.lexsort((q1,q2,q3))
for i in ind:
    print(q3[i],q2[i],q1[i])
```