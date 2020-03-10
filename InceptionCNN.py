import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ELU
class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(1,10), stride=(1,1), padding=(0,0))
        self.conv0_bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,7), stride=(1,1), padding=(0,3))
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,5), stride=(1,1), padding=(0,2))
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,1), stride=(1,1))
        self.conv4_bn = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,10), stride=(1,1), padding=(0,0))
        self.conv5_bn = nn.BatchNorm2d(128)
        
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,7), stride=(1,1), padding=(0,3))
        self.conv6_bn = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,5), stride=(1,1), padding=(0,2))
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv8_bn = nn.BatchNorm2d(128)        
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv9_bn = nn.BatchNorm2d(128)
        
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(1,10), stride=(1,1), padding=(0,0))
        self.conv10_bn = nn.BatchNorm2d(1024)
        
        self.conv11 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1,7), stride=(1,1), padding=(0,3))
        self.conv11_bn = nn.BatchNorm2d(1024)
        self.conv12 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1,5), stride=(1,1), padding=(0,2))
        self.conv12_bn = nn.BatchNorm2d(1024)
        self.conv13 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv13_bn = nn.BatchNorm2d(1024)        
        self.conv14 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv14_bn = nn.BatchNorm2d(1024)
        
        self.relu = nn.ELU(inplace=True) #nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(in_features=4096,  out_features=4)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,4), stride=(1,4), padding=(0,0))
        self.avgpool = nn.AvgPool2d(kernel_size=(1,12), stride=(1,12), padding=(0,0))
        
        
        
    def forward(self, X):
        # Nx1x22x1000
        xi = X.permute(0,2,1,3)
        x = self.conv0(xi)
        x = self.conv0_bn(x)
        x = self.relu(x)
        xi = self.maxpool(x)
        
        x = self.conv1(xi)
        x = self.conv1_bn(x)
        x1 = self.relu(x)
        x = self.conv2(xi)
        x = self.conv2_bn(x)
        x2 = self.relu(x)
        x = self.conv3(xi)
        x = self.conv3_bn(x)
        x3 = self.relu(x)
        x = self.conv4(xi)
        x = self.conv4_bn(x)
        x4 = self.relu(x)
        x = torch.cat((x1,x2,x3,x4),1)
        
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.relu(x)
        xi = self.maxpool(x)
        
        x = self.conv6(xi)
        x = self.conv6_bn(x)
        x1 = self.relu(x)
        x = self.conv7(xi)
        x = self.conv7_bn(x)
        x2 = self.relu(x)
        x = self.conv8(xi)
        x = self.conv8_bn(x)
        x3 = self.relu(x)
        x = self.conv9(xi)
        x = self.conv9_bn(x)
        x4 = self.relu(x)
        x = torch.cat((x1,x2,x3,x4),1)
        
        x = self.conv10(x)
        x = self.conv10_bn(x)
        x = self.relu(x)
        xi = self.maxpool(x)
        
        x = self.conv11(xi)
        x = self.conv11_bn(x)
        x1 = self.relu(x)
        x = self.conv12(xi)
        x = self.conv12_bn(x)
        x2 = self.relu(x)
        x = self.conv13(xi)
        x = self.conv13_bn(x)
        x3 = self.relu(x)
        x = self.conv14(xi)
        x = self.conv14_bn(x)
        x4 = self.relu(x)
        x = torch.cat((x1,x2,x3,x4),1)
        
        x = self.avgpool(x)
        x = self.fc(x.squeeze())
        return x
