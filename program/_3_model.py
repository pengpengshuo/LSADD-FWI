import torch
import torch.nn as nn
from _1_network_module import ConvBlock,UpsampleBlock,ConvBlock_Last,BasicLayer,PatchUpsample


class FWI(nn.Module): #Define the traditional FWI model and limit the velocity value between vmin and vmax
    def __init__(self,v_init,vmin,vmax):
        super().__init__()
        self.vmin=vmin
        self.vmax=vmax
        self.v=nn.Parameter(torch.logit((v_init-vmin)/(vmax-vmin)))
    def forward(self):
        return torch.sigmoid(self.v)*(self.vmax-self.vmin)+self.vmin

class CNN_FWI(nn.Module): #Convolutional neural network reparameterization model,where input:10*864*288
    def __init__(self,dim1=10, dim2=8, dim3=16, dim4=32, dim5=64,dim6=128):
        super().__init__()
        self.conv1_1 = ConvBlock(dim1, dim2, 3, 2, 1) #8*432*144

        self.conv2_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv2_2 = ConvBlock(dim2, dim3, 3, 1, 1)  # 16*216*72

        self.conv3_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv3_2 = ConvBlock(dim3, dim4, 3, 1, 1)  # 32*108*36

        self.conv4_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv4_2 = ConvBlock(dim4, dim5, 3, 1, 1)  # 64*54*18

        self.conv5_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv5_2 = ConvBlock(dim5, dim6, 3, 1, 1)  # 128*27*9

        #decoder
        self.upsampleblock2_1=UpsampleBlock(dim6,dim5,5,1,2) #64*54*18
        self.upsampleblock2_2 = UpsampleBlock(dim5, dim4, 5, 1, 2) #32*108*36
        self.upsampleblock2_3 = UpsampleBlock(dim4, dim3, 5, 1, 2) #16*216*72
        self.upsampleblock2_4=UpsampleBlock(dim3,dim2,5,1,2) #8*432*144

        self.convlast1 = ConvBlock_Last(dim2, dim2,5,1,2)
        self.convlast2 = ConvBlock_Last(dim2, 1, 3, 1, 1)

    def forward(self, x):
        x=self.conv1_1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x=self.conv4_1(x)
        x=self.conv4_2(x)
        x=self.conv5_1(x)
        x=self.conv5_2(x)
        #decoder2
        x = self.upsampleblock2_1(x)
        x = self.upsampleblock2_2(x)
        x= self.upsampleblock2_3(x)
        x=self.upsampleblock2_4(x)

        x = self.convlast1(x)
        x = self.convlast2(x)
        return x

class LSADD_FWI(nn.Module): #Proposed reparameterized model;input:10*864*192
    def __init__(self,dim1=10, dim2=8, dim3=16, dim4=32, dim5=64,dim6=128, num_heads=4, depth=2,window_size=6):
        super().__init__()
        self.conv1_1 = ConvBlock(dim1, dim2, 3, 2, 1) #8*432*96

        self.conv2_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv2_2 = ConvBlock(dim2, dim3, 3, 1, 1)  # 16*216*48

        self.conv3_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv3_2 = ConvBlock(dim3, dim4, 3, 1, 1)  # 32*108*24

        self.conv4_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv4_2 = ConvBlock(dim4, dim5, 3, 1, 1)  # 64*54*12

        self.conv5_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv5_2 = ConvBlock(dim5, dim6, 3, 1, 1)  # 128*27*6

        #decoder1
        self.sw_upsample1 = UpsampleBlock(dim6, dim5, 5,1,2)  # 64*54*12

        self.ps1_1 = PatchUpsample(dim5)
        self.swin1_1 = BasicLayer(dim4, (108, 24), depth, num_heads, window_size)

        self.ps1_2 = PatchUpsample(dim4)
        self.swin1_2 = BasicLayer(dim3, (216, 48), depth, num_heads, window_size)

        self.ps1_3 = PatchUpsample(dim3)
        self.swin1_3 = BasicLayer(dim2, (432, 96), depth, num_heads, window_size)
        #decoder2
        self.upsampleblock2_1_1 = nn.Upsample(scale_factor=(2, 1), mode='bilinear')
        self.upsampleblock2_1_2 = ConvBlock(dim6,dim5,5,1,2) # 64*54*6
        self.upsampleblock2_2 = UpsampleBlock(dim5, dim4, 5, 1, 2)
        self.upsampleblock2_3 = UpsampleBlock(dim4, dim3, 5, 1, 2)
        self.upsampleblock2_4=UpsampleBlock(dim3,dim2,5,1,2)

        self.convlast1 = ConvBlock_Last(dim2, dim2,5,1,2)
        self.convlast2 = ConvBlock_Last(dim2, 1, 3, 1, 1)

    def forward(self, x):
        x=self.conv1_1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x=self.conv4_1(x)
        x=self.conv4_2(x)
        x=self.conv5_1(x)
        x=self.conv5_2(x)
        #decoder1
        x1=self.sw_upsample1(x)

        x1 = self.ps1_1(x1)
        x1 = self.swin1_1(x1)

        x1 = self.ps1_2(x1)
        x1 = self.swin1_2(x1)

        x1 = self.ps1_3(x1)
        x1 = self.swin1_3(x1)
        #decoder2
        x2 = self.upsampleblock2_1_1(x)
        x2=self.upsampleblock2_1_2(x2)
        x2 = self.upsampleblock2_2(x2)
        x2 = self.upsampleblock2_3(x2)
        x2=self.upsampleblock2_4(x2)

        x = torch.cat([x1, x2], dim=3)  # Dim=1 indicates concatenation along the second dimension

        x = self.convlast1(x)
        x = self.convlast2(x)
        return x

class LSADD_FWI_temp(nn.Module): #A reparameterized model with attention adjustment factor;input:10*864*192
    def __init__(self,dim1=10, dim2=8, dim3=16, dim4=32, dim5=64,dim6=128, num_heads=4, depth=2,window_size=6,temp=2):
        super().__init__()
        self.conv1_1 = ConvBlock(dim1, dim2, 3, 2, 1) #8*432*96

        self.conv2_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv2_2 = ConvBlock(dim2, dim3, 3, 1, 1)  # 16*216*48

        self.conv3_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv3_2 = ConvBlock(dim3, dim4, 3, 1, 1)  # 32*108*24

        self.conv4_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv4_2 = ConvBlock(dim4, dim5, 3, 1, 1)  # 64*54*12

        self.conv5_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        self.conv5_2 = ConvBlock(dim5, dim6, 3, 1, 1)  # 128*27*6

        #decoder1
        self.sw_upsample1 = UpsampleBlock(dim6, dim5, 5,1,2)  # 64*54*12

        self.ps1_1 = PatchUpsample(dim5)
        self.swin1_1 = BasicLayer(dim4, (108, 24), depth, num_heads, window_size,qk_scale=1/(temp*2*1.4))

        self.ps1_2 = PatchUpsample(dim4)
        self.swin1_2 = BasicLayer(dim3, (216, 48), depth, num_heads, window_size,qk_scale=1/(temp*2))

        self.ps1_3 = PatchUpsample(dim3)
        self.swin1_3 = BasicLayer(dim2, (432, 96), depth, num_heads, window_size,qk_scale=1/(temp*1.4))
        #decoder2
        self.upsampleblock2_1_1 = nn.Upsample(scale_factor=(2, 1), mode='bilinear')
        self.upsampleblock2_1_2 = ConvBlock(dim6,dim5,5,1,2) # 64*54*6
        self.upsampleblock2_2 = UpsampleBlock(dim5, dim4, 5, 1, 2)
        self.upsampleblock2_3 = UpsampleBlock(dim4, dim3, 5, 1, 2)
        self.upsampleblock2_4=UpsampleBlock(dim3,dim2,5,1,2)

        self.convlast1 = ConvBlock_Last(dim2, dim2,5,1,2)
        self.convlast2 = ConvBlock_Last(dim2, 1, 3, 1, 1)

    def forward(self, x):
        x=self.conv1_1(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x=self.conv4_1(x)
        x=self.conv4_2(x)
        x=self.conv5_1(x)
        x=self.conv5_2(x)
        #decoder1
        x1=self.sw_upsample1(x)

        x1 = self.ps1_1(x1)
        x1 = self.swin1_1(x1)

        x1 = self.ps1_2(x1)
        x1 = self.swin1_2(x1)

        x1 = self.ps1_3(x1)
        x1 = self.swin1_3(x1)
        #decoder2
        x2 = self.upsampleblock2_1_1(x)
        x2=self.upsampleblock2_1_2(x2)
        x2 = self.upsampleblock2_2(x2)
        x2 = self.upsampleblock2_3(x2)
        x2=self.upsampleblock2_4(x2)

        x = torch.cat([x1, x2], dim=3)

        x = self.convlast1(x)
        x = self.convlast2(x)
        return x