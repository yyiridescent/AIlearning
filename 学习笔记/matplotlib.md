# <center>Matplotlib
## <center>普通图像
### 绘制图像
绘制基本函数图像
##### 取值
以sinx,cosx为例
```python
x=np.linspace(-np.pi,np.pi,256,endpoint=True)
cosx,sinx=np.cos(x),np.sin(x)
```
##### 绘制图像并显示
使用默认配置进行制图
```python
plt.plot(x,cosx)
plt.plot(x,sinx)

plt.show()
```

### 改变默认配置
```python
 #8*6的图；分辨率时80
plt.figure(figsize=(8,6),dpi=80)

#创建子图
plt.subplot(1,1,1)

#数据导入
x=np.linspace(-np.pi,np.pi,256,endpoint=True)
cosx,sinx=np.cos(x),np.sin(x)

#cosx图像绘制
plt.plot(x,cosx,color="blue",linewidth=1.0,linstyle='-')

#sinx图像绘制
plt.plot(x,sinx,color="red",linewidth=1.0,linstyle='-')

#设置横轴的上下限
plt.xlim(-4.0,4.0)

#设置横轴
plt.xticks(np.linspace(-4,4,9,endpoint=True))

#设置纵轴的上下限
plt.ylim(-1.0，1.0)

#设置横轴
plt.yticks(np.linspace(-1,1,5,endpoint=True))

#展示
plt.show()
```

### 设置图片边界
```python
xmin ,xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()

dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2

xlim(xmin - dx, xmax + dx)
ylim(ymin - dy, ymax + dy)
```

### 移动x轴和y轴
每幅图有四条脊柱，为了将x轴和y轴放在图的中间，需要4条脊柱中的两条（上和右）设置为无色，然后调整剩下的两条到合适的位置，定为数据空间的0点。

```python
ax = plt.gca()
#右边和上面的脊柱“消失”
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

#左边和下面的脊柱移位置
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))
```

### 增加图例
```python
plt.plot(x,cosx,coloe="blue", linewidth=2.5,linestyle="-", label="cosx")
plt.plot(x,cosx,coloe="red",linewidth=2.5,linestyle="-", label="sinx")

plt.legend(loc='upper left')#左上角
plt.show()
```

### 给特殊点注释
```python
t = 2*np.pi/3
plt.plot([t,t],[0,np.cos(t)], color ='blue', linewidth=2.5, linestyle="--")
plt.scatter([t,],[np.cos(t),], 50, color ='blue')

plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
         xy=(t, np.sin(t)), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.plot([t,t],[0,np.sin(t)], color ='red', linewidth=2.5, linestyle="--")
plt.scatter([t,],[np.sin(t),], 50, color ='red')

plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
         xy=(t, np.cos(t)), xycoords='data',
         xytext=(-90, -50), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
```

## <center>其他图像
部分图像代码示例
#### 散点图
```python
from matplotlib.pylab import plt
import numpy as np

n = 1024
X = np.random.normal(0,1,n)
Y = np.random.normal(0,1,n)

plt.scatter(X,Y)
plt.show()
```

#### 条形图
```python
from matplotlib.pylab import plt
import numpy as np

n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')

for x,y in zip(X,Y1):
    plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

plt.ylim(-1.25,+1.25)
plt.show()
```

#### 等高线图
```python
from matplotlib.pylab import plt
import numpy as np

def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

plt.contourf(X, Y, f(X,Y), 8, alpha=.75, cmap='jet')
C = plt.contour(X, Y, f(X,Y), 8, colors='black', linewidth=.5)
plt.show()
```

#### 灰度图
```python
from matplotlib.pylab import plt
import numpy as np

def f(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 10
x = np.linspace(-3,3,4*n)
y = np.linspace(-3,3,3*n)
X,Y = np.meshgrid(x,y)
plt.imshow(f(X,Y))

plt.show()

```

#### 饼状图
```python
from matplotlib.pylab import plt
import numpy as np

n = 20
Z = np.random.uniform(0,1,n)
plt.pie(Z)
plt.show()
```

#### 3D图
```python
from matplotlib.pylab import plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = Axes3D(fig)
X = np.arange(-4, 4, 0.25)
Y = np.arange(-4, 4, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot')

plt.show()
```