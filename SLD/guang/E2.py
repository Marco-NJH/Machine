from scipy.constants import h
from scipy.constants import Boltzmann
import math
from scipy.integrate import quad
import pandas as pd
data = pd.read_excel('计算发射.xls', sheet_name='Sheet1', usecols=[0, 1], header=0, nrows=356)
lamda = data.iloc[:, 0].values
R = data.iloc[:, 1].values
e1 = math.e
c = 299792458
ad = c**2
Ea = 2*h*(c**2)
A = 0
B = 0
for i in range(355):
    long = (lamda[i]/10**6)**5
    zhi = (h*c)/((lamda[i]/10**6)*Boltzmann*300)
    ds = (lamda[i+1]-lamda[i])*(Ea/(long*(e1**zhi-1)))
    A += ds
for i in range(355):
    long = (lamda[i] / 10 ** 6) ** 5
    zhi = (h * c) / ((lamda[i] / 10 ** 6) * Boltzmann * 300)
    ds = (lamda[i+1]-lamda[i])*(Ea/(long*(e1**zhi-1)))*(R[i]/100)
    B+=ds

result = B/A
print(result)
# def f(x):
#     return Ea/(x**5)*(e1**((h*c)/x*Boltzmann*300)-1)


# print(e)
# print(Ea)
# print(h)
# print(Boltzmann)