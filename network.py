import torch
from torch.nn.utils import weight_norm as wn
import torch.nn as nn

def downshuffle(var,r):
        b,c,h,w = var.size()
        out_channel = c*(r**2)
        out_h = h//r
        out_w = w//r
        return var.contiguous().view(b, c, out_h, r, out_w, r).permute(0,1,3,5,2,4).contiguous().view(b,out_channel, out_h, out_w).contiguous()

def conv_layer(inc, outc, kernel_size=3, groups=1, bias=False, negative_slope=0.2, bn=False, init_type='normal', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=4, num_classes=3, weight_normalization = True):

    layers = []
    
    if bn:
        m = nn.BatchNorm2d(inc)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        layers.append(m)
    
    if activation=='before':
        layers.append(nn.ReLU())
        #layers.append(nn.LeakyReLU(negative_slope=negative_slope))
        
    if pixelshuffle_init:
        m = nn.Conv2d(in_channels=inc, out_channels=num_classes * (upscale ** 2),
                                  kernel_size=3, padding = 3//2, groups=1, bias=True, stride=1)
        nn.init.constant_(m.bias, 0)
        with torch.no_grad():
            kernel = ICNR(m.weight, upscale, negative_slope, fan_type)
            m.weight.copy_(kernel)
    else:
        m = nn.Conv2d(in_channels=inc, out_channels=outc,
     kernel_size=kernel_size, padding = (kernel_size-1)//2, groups=groups, bias=bias, stride=1)
        init_gain = 0.02
        if init_type == 'normal':
            torch.nn.init.normal_(m.weight, 0.0, init_gain)
        elif init_type == 'xavier':
            torch.nn.init.xavier_normal_(m.weight, gain = init_gain)
        elif init_type == 'kaiming':
            torch.nn.init.kaiming_normal_(m.weight, mode=fan_type, nonlinearity='relu')
#            torch.nn.init.kaiming_normal_(m.weight, a=negative_slope, mode=fan_type, nonlinearity='leaky_relu')
        elif init_type == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight)
        else:
            raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
    if weight_normalization:
        layers.append(wn(m))
    else:
        layers.append(m)
        print('Weight normalisation NOT done !')
    
    if activation=='after':
        layers.append(nn.ReLU())
            #layers.append(nn.LeakyReLU(negative_slope=negative_slope))
            
    return nn.Sequential(*layers)
    
class ResBlock(nn.Module):    
    def __init__(self,n,in_c,k=3):
        super(ResBlock, self).__init__()
        
        layers = []
        for idx in range(n-1):
            layers.append(conv_layer(in_c, in_c, kernel_size=k, groups=1, bias=True, negative_slope=0.2, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True))
        
        self.layers = nn.Sequential(*layers)
        
        self.conv = conv_layer(in_c, in_c, kernel_size=k, groups=1, bias=True, negative_slope=1, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)

    def forward(self, x):
        return self.conv(self.layers(x)) + x
        
class stageIII(nn.Module):    
    def __init__(self):
        super(stageIII, self).__init__()
        
        self.up2 = nn.PixelShuffle(2)
        
        ipchannels = int((9+1+1)*3)
        
        self.resblk1 = ResBlock(2,ipchannels)
        self.resblk2 = ResBlock(4,int(ipchannels*2*4))
        self.resblk3 = ResBlock(2,ipchannels*2*2)
        
        self.conv2 = conv_layer(inc=int(ipchannels*2*2), outc=int(9*3), kernel_size=3, groups=1, bias=True, negative_slope=1.0, bn=False, init_type='kaiming', fan_type='fan_in', activation=False, pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)

    def forward(self, ip):
        x = self.resblk1(ip)
        return self.conv2(self.resblk3(torch.cat((self.up2(self.resblk2(torch.cat((downshuffle(x,2),downshuffle(ip,2)),dim=1))),x,ip),dim=1)))
        
class DisparityModule(nn.Module):
    def __init__(self):
      super(DisparityModule, self).__init__()
      self.fc0 = nn.Linear(2, 9)
      self.fc1 = nn.Linear(9, 9)
      self.fc2 = nn.Linear(9, 9)
      self.fc3 = nn.Linear(9, 9)
      self.fc4 = nn.Linear(9, 9)
      self.fc5 = nn.Linear(9, 9)
      self.fc6 = nn.Linear(9, 9)
      self.fc7 = nn.Linear(9, 9)
      self.fc8 = nn.Linear(9, 9)
      self.fc9 = nn.Linear(9, 9)
      self.fc10 = nn.Linear(9, 9)
      self.fc11 = nn.Linear(9, 9)
      self.fc12 = nn.Linear(9, 9)
      self.fc13 = nn.Linear(9, 9)
      self.fc14 = nn.Linear(9, 9)
      self.fc15 = nn.Linear(9, 9)
      self.fc16 = nn.Linear(9, 9)
      self.fc17 = nn.Linear(9, 9)
      self.fc18 = nn.Linear(9, 9)
      self.bias = nn.Linear(9,3)
      self.relu = nn.ReLU(inplace=False)
    
    def forward(self, disparity):
      row1 = self.fc1(self.fc0(disparity))
      row2 = self.fc2(self.relu(row1))
      row3 = self.fc3(self.relu(row2))
      row4 = self.fc4(self.relu(row3))
      row5 = self.fc5(self.relu(row4))
      row6 = self.fc6(self.relu(row5))
      row7 = self.fc7(self.relu(row6))
      row8 = self.fc8(self.relu(row7))
      row9 = self.fc9(self.relu(row8))
      row10 = self.fc10(self.relu(row9))
      row11 = self.fc11(self.relu(row10))
      row12 = self.fc12(self.relu(row11))
      row13 = self.fc13(self.relu(row12))
      row14 = self.fc14(self.relu(row13))
      row15 = self.fc15(self.relu(row14))
      row16 = self.fc16(self.relu(row15))
      row17 = self.fc17(self.relu(row16))
      row18 = self.fc18(self.relu(row17))
      bias = self.bias(self.relu(row18))
      return torch.cat((row1,row2,row3,row4,row5,row6,row7,row8,row9,row10,row11,row12,row13,row14,row15,row16,row17,row18),dim=1).reshape((3, 6, 3, 3)), bias.reshape((3))

        
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        self.stagIII = stageIII()
        self.mlp = DisparityModule()
        
        self.conv1 = conv_layer(81, 64, kernel_size=3, groups=1, bias=True, negative_slope=0.0, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        self.conv2 = conv_layer(64, 3, kernel_size=3, groups=1, bias=True, negative_slope=0.0, bn=False, init_type='kaiming', fan_type='fan_in', activation='after', pixelshuffle_init=False, upscale=False, num_classes=False, weight_normalization = True)
        
        
    def forward(self,low):
        b, s, t, c, h, w = low.shape
        avg = self.conv2(self.conv1(low[:,:,:,1,:,:].clone().flatten(start_dim=1, end_dim=2)))
        out = low.clone()
        
        for row in [1,4,7]:
            for col in [1,4,7]:
                weight,bias = self.mlp(torch.tensor([[row-4,col-4]], dtype=torch.float32, requires_grad=False))#.to(torch.device("cuda")))
                inter = torch.nn.functional.conv2d(torch.cat((avg, low[:,row,col,:,:,:]), dim=1), weight, bias=bias, stride=1, padding=1, dilation=1, groups=1)
                sai_neighbours = torch.cat((low[:,row-1,col-1,:,:,:], low[:,row-1,col,:,:,:], low[:,row-1,col+1,:,:,:], low[:,row,col-1,:,:,:], low[:,row,col,:,:,:], low[:,row,col+1,:,:,:], low[:,row+1,col-1,:,:,:], low[:,row+1,col,:,:,:],low[:,row+1,col+1,:,:,:],inter,avg),dim=1)
                out[:,row-1:row+2,col-1:col+2,:,:,:] = self.stagIII(sai_neighbours).view(b,3,3,c,h,w)
                
        return torch.clamp(out,min=0.0, max=1.0)



