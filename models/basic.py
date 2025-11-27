#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch.nn as nn
import torch.nn.functional as F
import copy
# In[1]:

class FC2(nn.Module):
    def __init__(self):
        super(FC2, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):

        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        return x

class CNN(nn.Module):
    def __init__(self,num_classes = 10,l2_norm = False):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5,padding=1)
        self.fc1 = nn.Linear(64 * 6 * 6, 394)
        self.fc2 = nn.Linear(394, 192)
        self.fc3 = nn.Linear(192, num_classes)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = (self.fc3(x))
        return x

class CNN_dropout(nn.Module):
    def __init__(self):
        super(CNN_dropout, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5,padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 64, 5,padding=1)
        self.dropout1 = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(64 * 6 * 6, 394)
        self.fc2 = nn.Linear(394, 192)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64*6*6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = (self.fc3(x))
        return x
      

# In[ ]:




