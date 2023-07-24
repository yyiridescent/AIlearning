import math
import numpy as np

N=input()
A=[int(n) for n in N.split()]
# for i in range(1,n+1):
#     a=int(input())
#     A.append(a)
# print(a)


n=np.array(A).shape[0]

max_num=np.amax(A)

sum=0
for j in range(0,n):
    sum= sum + A[j]


result=math.e**(max_num/sum)

print(result)