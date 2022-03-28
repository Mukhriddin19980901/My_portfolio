#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import torch
import matplotlib.pyplot as plt


# In[11]:


data= np.loadtxt(r"C:/Users/USER/Datasets2021/heart.csv",delimiter=',',dtype=np.float32)  
data_x = torch.from_numpy(data[:305,:13])
data_y = torch.from_numpy(data[:305,[13]])


# In[12]:


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(13,11)
        self.linear2 = torch.nn.Linear(11,9)
        self.linear3 = torch.nn.Linear(9,7)
        self.linear4 = torch.nn.Linear(7,5)
        self.linear5 = torch.nn.Linear(5,3)
        self.linear6 = torch.nn.Linear(3,1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self,x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        x = self.linear6(x)
        y_pred = torch.sigmoid(x)
        return y_pred
model = Model()


# In[40]:


criteria = torch.nn.MSELoss(reduction = "mean")
optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)
epochs=500
loss=[]
acc=[]
for epoch in range(epochs+1):
    y_bash = model(data_x)
    xato = criteria(y_bash,data_y)
    if epoch % 100 == 0:
        print(f"Epoch : {epoch} : Loss : {xato.item()} Accuracy : {1-xato.item()}")
    loss.append(xato.item())
    acc.append(1-xato.item())
    optimizer.zero_grad()
    xato.backward()
    optimizer.step()
print(f"Model evaluate  : {[xato.item(), 1-xato.item()]}")


# In[41]:


plt.plot(loss,label='loss')
plt.plot(acc,label="aniqlik")
plt.title("heart attack risk")
plt.xlabel("xatolik")
plt.legend(loc = 'best')


# In[27]:


x_test = torch.Tensor([42,1,0,136,315,0,1,125,1,1.8,1,0,1])
print(model(x_test))
if model(x_test)>0.5:
    print("Positive risk")
else:
    print("not risk")
y_test = torch.Tensor([66,0,2,146,278,0,0,152,0,0,1,1,2])
print(model(y_test))
if model(y_test)<0.5:
    print("not risk")
else:
    print('Positive risk')


# In[ ]:




