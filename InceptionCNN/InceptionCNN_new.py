import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, ELU
class InceptionNet(nn.Module):
    def __init__(self):
        super(InceptionNet, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(1,9), stride=(1,1), padding=(0,4))
        self.conv0_bn = nn.BatchNorm2d(32)
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(1,7), stride=(1,1), padding=(0,3))
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(1,5), stride=(1,1), padding=(0,2))
        self.conv2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=22, out_channels=32, kernel_size=(1,1), stride=(1,1))
        self.conv4_bn = nn.BatchNorm2d(32)
        
        self.conv5 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,9), stride=(1,1), padding=(0,4))
        self.conv5_bn = nn.BatchNorm2d(64)       
        self.conv6 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,7), stride=(1,1), padding=(0,3))
        self.conv6_bn = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,5), stride=(1,1), padding=(0,2))
        self.conv7_bn = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv8_bn = nn.BatchNorm2d(64)        
        self.conv9 = nn.Conv2d(in_channels=160, out_channels=64, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv9_bn = nn.BatchNorm2d(64)
        
        self.conv10 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(1,9), stride=(1,1), padding=(0,4))
        self.conv10_bn = nn.BatchNorm2d(128)       
        self.conv11 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(1,7), stride=(1,1), padding=(0,3))
        self.conv11_bn = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(1,5), stride=(1,1), padding=(0,2))
        self.conv12_bn = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(1,3), stride=(1,1), padding=(0,1))
        self.conv13_bn = nn.BatchNorm2d(128)        
        self.conv14 = nn.Conv2d(in_channels=320, out_channels=128, kernel_size=(1,1), stride=(1,1), padding=(0,0))
        self.conv14_bn = nn.BatchNorm2d(128)
        
        self.relu = nn.ELU(inplace=True) 
        
        self.fc = nn.Linear(in_features=640,  out_features=4)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,10), stride=(1,5), padding=(0,0))
        self.avgpool = nn.AvgPool2d(kernel_size=(1,38), stride=(1,38), padding=(0,0))
        
        
        
    def forward(self, X):
        # Nx1x22x1000
        xi = X.permute(0,2,1,3)
        x = self.conv0(xi)
        x = self.conv0_bn(x)
        x0 = self.relu(x)
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
        x = torch.cat((x0,x1,x2,x3,x4),1)
        
        xi = self.maxpool(x)
                
        x = self.conv5(xi)
        x = self.conv5_bn(x)
        x0 = self.relu(x)
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
        x = torch.cat((x0,x1,x2,x3,x4),1)
       
        xi = self.maxpool(x)

        x = self.conv10(xi)
        x = self.conv10_bn(x)
        x0 = self.relu(x)
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
        x = torch.cat((x0,x1,x2,x3,x4),1)

        x = self.avgpool(x)
        x = self.fc(x.squeeze())
        return x