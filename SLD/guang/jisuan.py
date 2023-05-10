import pandas as pd

# 公式E1
data1 = pd.read_excel('计算反射.xlsx', sheet_name='Sheet2', usecols=[0, 1], header=0, nrows=441)
data2 = pd.read_excel('计算反射.xlsx', sheet_name='Sheet2', usecols=[2, 3], header=0, nrows=441)
lamda1 = data1.iloc[:, 0].values
lamda2 = data2.iloc[:,0].values
R = data1.iloc[:, 1].values
I = data2.iloc[:, 1].values

A = 0  # I积分
B = 0  # R*I积分
for i in range(440):
    ds = (lamda2[i + 1] - lamda2[i]) * I[i]
    A += ds

F = R * I
for i in range(440):
    ds = (lamda1[i + 1] - lamda1[i]) * F[i]
    B += ds

R_averge = B / A
print(R_averge)