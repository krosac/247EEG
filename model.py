import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ELU
from torch.optim import Adam, SGD

class Net_lstm(nn.Module):
    def __init__(self):
        super(Net_lstm, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,10),  stride=(1,1))
        self.conv1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(21,1), stride=(1,1))
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,10), stride=(1,1))
        self.conv3_bn = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1,10), stride=(1,1))
        self.conv4_bn = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,0))
        self.relu = nn.ELU() 
        self.lstm1 = nn.LSTM(128*24, 128, 1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, 1, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(128, 32, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(in_features=64,  out_features=4)
        self.dropout = nn.Dropout(0.4)
        self.softmax = nn.Softmax()
        
        
        
    def forward(self, X):
        # Nx1x22x1000
        x = self.conv1(X)
        x = self.relu(x)
        x = self.conv1_bn(x) 
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv2_bn(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv3_bn(x)
        x = self.maxpool(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv4_bn(x)
        x = self.maxpool(x)
        x = x.permute(0,3,1,2) # important! -- without <50% 128x2x12->12x128x2
        x = x.reshape(-1,1,12*128*2)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x = x.squeeze()
        x = self.fc(x)
        return x
        