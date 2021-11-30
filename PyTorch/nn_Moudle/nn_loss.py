import torch
from torch.nn import L1Loss, MSELoss

input = torch.tensor([1,2,3],dtype=torch.float32)
target = torch.tensor([1,2,5],dtype=torch.float32)

input = torch.reshape(input,(1,1,1,3))
target = torch.reshape(target,(1,1,1,3))

loss = L1Loss()
result = loss(input,target)

loss_mse = MSELoss()
result2 = loss_mse(input,target)

print(result)
print(result2)