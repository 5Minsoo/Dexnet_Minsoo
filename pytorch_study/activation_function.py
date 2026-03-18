import torch

x_train=torch.FloatTensor([1,2,3])
y_train=torch.FloatTensor([4,5,6])
print(x_train.shape)

W=torch.zeros(x_train.shape,requires_grad=True)
b=torch.zeros(x_train.shape,requires_grad=True)
print(W)
