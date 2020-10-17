import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
from torchvision import transforms

# Hyper parameter
EPOCH = 1
BATCH_SIZE = 1
LR = 0.001

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.conv(x)

class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=2, output_padding=1):
        super().__init__()
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=padding, output_padding=output_padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.upconv(x)

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=2 if downsample else 1),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=out_channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x + residual
        return self.relu(x)

class MultiScaleConv(nn.Module):
    def __init__(self, kernel_1, kernel_2, kernel_3, kernel_4, dilation_1, dilation_2, dilation_3, dilation_4, channels):
        super().__init__()
        out_channels = channels // 4
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=kernel_1, dilation=dilation_1, padding=self._padding_calculator(dilation_1, kernel_1))
        self.conv_2 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=kernel_2, dilation=dilation_2, padding=self._padding_calculator(dilation_2, kernel_2))
        self.conv_3 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=kernel_3, dilation=dilation_3, padding=self._padding_calculator(dilation_3, kernel_3))
        self.conv_4 = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=kernel_4, dilation=dilation_4, padding=self._padding_calculator(dilation_4, kernel_4))

    def forward(self, x):
        return torch.cat((self.conv_1(x), self.conv_2(x), self.conv_3(x), self.conv_4(x)), dim=1)

    def _padding_calculator(self, dilation, kernel):
        return (dilation * (kernel - 1)) // 2

class NuClick(nn.Module):
    # Input size is 128x128
    def __init__(self):
        super().__init__()

        # General
        self.max_pool = nn.MaxPool2d(kernel_size=2)     

        # Block 1
        self.conv_1 = ConvBlock(3, 64, 7, 3)             
        self.conv_2 = ConvBlock(64, 32, 5, 2)            
        self.conv_3 = ConvBlock(32, 32, 3, 1)            

        # Block 2
        self.resnet_1 = Residual(32, 64)                
        self.resnet_2 = Residual(64, 64)

        # Block 3
        self.resnet_3 = Residual(64, 128)
        self.multi_1 = MultiScaleConv(3, 3, 5, 5, 1, 3, 3, 6, 128)
        self.resnet_4 = Residual(128, 128)

        # Block 4
        self.resnet_5 = Residual(128, 256)
        self.resnet_6 = Residual(256, 256)

        # Block 5
        self.resnet_7 = Residual(256, 512)
        self.resnet_8 = Residual(512, 512)
        self.upconv_1 = UpConvBlock(512, 512, 2)

        # Block 6
        self.resnet_9 = Residual(768, 256)
        self.multi_2 = MultiScaleConv(3, 3, 5, 5, 1, 3, 2, 3, 256)
        self.resnet_10 = Residual(256, 256)
        self.upconv_2 = UpConvBlock(256, 256)

        # Block 7
        self.resnet_11 = Residual(384, 128)
        self.resnet_12 = Residual(128, 128)
        self.upconv_3 = UpConvBlock(128, 128)

        # Block 8
        self.resnet_13 = Residual(192, 64)
        self.multi_3 = MultiScaleConv(3, 3, 5, 7, 1, 3, 3, 3, 64)
        self.resnet_14 = Residual(64, 64)
        self.upconv_4 = UpConvBlock(64, 64)

        # Block 9
        self.conv_4 = ConvBlock(96, 64, 3, 1)
        self.conv_5 = ConvBlock(64, 32, 3, 1)
        self.conv_6 = ConvBlock(32, 32, 3, 1)
        self.conv_7 = ConvBlock(32, 1, 1)


    def forward(self, x):
        # Block 1
        x = self.conv_1(x)                              # 64 x 128 x 128    (pad: 3)
        x = self.conv_2(x)                              # 32 x 128 x 128    (pad: 2)
        node_1 = self.conv_3(x)                         # 32 x 128 x 128    (pad: 1)
        
        # Block 2
        x = self.max_pool(node_1)                       # 32 x 64 x 64
        x = self.resnet_1(x)                            # 64 x 64 x 64
        node_2 = self.resnet_2(x)                       # 64 x 64 x 64

        # Block 3
        x = self.max_pool(node_2)                       # 64 x 32 x 32
        x = self.resnet_3(x)                            # 128 x 32 x 32
        x = self.multi_1(x)                             # 128 x 32 x 32
        node_3 = self.resnet_4(x)                       # 128 x 32 x 32

        # Block 4
        x = self.max_pool(node_3)                       # 128 x 16 x 16
        x = self.resnet_5(x)                            # 256 x 16 x 16
        node_4 = self.resnet_6(x)                       # 256 x 16 x 16

        # Block 5
        x = self.max_pool(node_4)                       # 256 x 8 x 8
        x = self.resnet_7(x)                            # 512 x 8 x 8
        x = self.resnet_8(x)                            # 512 x 8 x 8
        node_5 = self.upconv_1(x)                       # 512 x 16 x 16     (pad: 2)

        # Block 6
        x = torch.cat((node_4, node_5), dim=1)          # 768 x 16 x 16
        x = self.resnet_9(x)                            # 256 x 16 x 16
        x = self.multi_2(x)                             # 256 x 16 x 16
        x = self.resnet_10(x)                           # 256 x 16 x 16
        node_6 = self.upconv_2(x)                       # 256 x 32 x 32     (pad: 2)

        # Block 7
        x = torch.cat((node_3, node_6), dim=1)          # 384 x 32 x 32
        x = self.resnet_11(x)                           # 128 x 32 x 32
        x = self.resnet_12(x)                           # 128 x 32 x 32
        node_7 = self.upconv_3(x)                       # 128 x 64 x 64     (pad: 2)

        # Block 8
        x = torch.cat((node_2, node_7), dim=1)          # 192 x 64 x 64
        x = self.resnet_13(x)                           # 64 x 64 x 64
        x = self.multi_3(x)                             # 64 x 64 x 64
        x = self.resnet_14(x)                           # 64 x 64 x 64
        node_8 = self.upconv_4(x)                       # 64 x 128 x 128    (pad: 2)

        # Block 9
        x = torch.cat((node_1, node_2), dim=1)          # 96 x 128 x 128
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.conv_7(x)
        
model = NuClick()
print(model)