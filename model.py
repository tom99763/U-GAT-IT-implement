import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.conv=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=(3, 3), stride=(1, 1),bias=False),
            nn.InstanceNorm2d(num_features=channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=(3, 3), stride=(1, 1),bias=False),
            nn.InstanceNorm2d(num_features=channels),
        )

    def forward(self,x):
        return x+self.conv(x)


class Adapolin_Stack(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.act=nn.ReLU(inplace=True)
        self.conv1=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=(3, 3), stride=(1, 1),bias=False),
        )

        self.conv2=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=(3, 3), stride=(1, 1),bias=False)
        )
        self.adapolin1=AdaLIN(channels=channels,eat=True)
        self.adapolin2=AdaLIN(channels=channels,eat=True)
    def forward(self,x,gamma=None,beta=None):
        o=self.conv1(x)
        o=self.adapolin1(o,gamma,beta)
        o=self.act(o)
        o=self.conv2(o)
        o=self.adapolin2(o,gamma,beta)
        return o+x




class AdaLIN(nn.Module):
    def __init__(self,channels,eat=False):
        super().__init__()
        self.eat=eat
        self.In=nn.InstanceNorm2d(num_features=channels)
        self.rho = nn.Parameter(data=torch.Tensor(1, channels, 1, 1))

        if not self.eat:
            self.gamma=nn.Parameter(data=torch.Tensor(1,channels,1,1))
            self.beta = nn.Parameter(data=torch.Tensor(1, channels, 1, 1))
            self.gamma.data.fill_(1.0)
            self.beta.data.fill_(0.0)
            self.rho.data.fill_(0.0)
        else:
            self.rho.data.fill_(0.9)


    def forward(self,x,gamma=None,beta=None):
        In=self.In(x)
        Ln=(x-torch.mean(x,dim=[1,2,3],keepdim=True))/(torch.sqrt(torch.var(x,dim=[1,2,3],keepdim=True))+1e-5)
        x=self.rho*In+(1-self.rho)*Ln
        if self.eat:
            x=gamma*x+beta
        else:
            x=self.gamma*x+self.beta
        return x

'''
x=torch.randn(1,256,32,32)
gamma=torch.randn(1,256,1,1)
beta=torch.randn(1,256,1,1)
ada=Adapolin_Stack(256)
res=ResBlock(256)
print(ada(x,gamma,beta).shape)
print(res(x).shape)
'''
class Generator(nn.Module):
    def __init__(self,base=64,n_res=4,downsample=2):
        super().__init__()
        self.downsample=downsample
        self.base=base
        self.n_res=n_res
        self.down=[]
        self.up=[]

        self.down+=[nn.ReflectionPad2d(3),
                    nn.Conv2d(in_channels=3,out_channels=self.base,
                              kernel_size=(7,7),
                              stride=(1,1),
                              bias=False),
                    nn.InstanceNorm2d(self.base),
                    nn.ReLU(inplace=True)
                    ]

        for i in range(self.downsample):
            self.scale=2**(i)
            self.down+=[nn.ReflectionPad2d(1),
                    nn.Conv2d(in_channels=self.base*self.scale,out_channels=self.base*self.scale*2,
                              kernel_size=(3,3),
                              stride=(2,2),
                              bias=False),
                    nn.InstanceNorm2d(self.base*2),
                    nn.ReLU(inplace=True)
                    ]

        self.res=[ResBlock(self.base*self.scale*2)  for _ in range(self.n_res)]


        self.gmp_fc=nn.Linear(in_features=self.base*self.scale*2,
                              out_features=1,bias=False)
        self.gap_fc = nn.Linear(in_features=self.base * self.scale * 2,
                                out_features=1, bias=False)

        self.conv11=nn.Conv2d(in_channels=self.base*self.scale*4,
                          out_channels=self.base*self.scale*2,
                          kernel_size=(1,1),
                          stride=(1,1))
        self.act=nn.ReLU(inplace=True)

        self.mlp=[nn.Linear(in_features=self.base*self.scale*2,
                              out_features=self.base*self.scale*2,bias=False),
                  nn.ReLU(inplace=True),
                  nn.Linear(in_features=self.base * self.scale * 2,
                            out_features=self.base * self.scale * 2, bias=False),
                  nn.ReLU(inplace=True)
                  ]
        self.gamma=nn.Linear(in_features=self.base * self.scale * 2,
                                out_features=self.base * self.scale * 2, bias=False)
        self.beta = nn.Linear(in_features=self.base * self.scale * 2,
                               out_features=self.base * self.scale * 2, bias=False)

        for i in range(self.n_res):
            setattr(self,f'adapolin_{i}',Adapolin_Stack(self.base*self.scale*2))


        for i in range(self.downsample):
            self.scale=2**(self.downsample-i-1)
            self.up+=[
                nn.Upsample(scale_factor=2),
                nn.ReflectionPad2d(1),
                nn.Conv2d(in_channels=self.base*self.scale*2,
                          out_channels=self.base*self.scale,
                          kernel_size=(3,3),
                          stride=(1,1),
                          bias=False),
                AdaLIN(channels=self.base*self.scale),
                nn.ReLU(inplace=True)
            ]
        self.up+=[nn.ReflectionPad2d(3),
                  nn.Conv2d(in_channels=self.base,
                            out_channels=3,
                            kernel_size=(7, 7),
                            stride=(1, 1),
                            bias=False),
                  nn.Tanh()]

        self.down=nn.Sequential(*self.down)
        self.up=nn.Sequential(*self.up)
        self.res=nn.Sequential(*self.res)
        self.mlp=nn.Sequential(*self.mlp)

    def forward(self,x):
        x=self.down(x)
        x=self.res(x)
        gap,gap_logit=self.class_activation_map(x,
                            pool_fn=F.adaptive_avg_pool2d,
                            map_fn=self.gap_fc)
        gmp,gmp_logit=self.class_activation_map(x,
                            pool_fn=F.adaptive_max_pool2d,
                            map_fn=self.gmp_fc)

        x=torch.cat([gap,gmp],dim=1)
        x=self.act(self.conv11(x))
        logit=torch.cat([gap_logit,gmp_logit],dim=1)
        heatmap=torch.sum(x,dim=1)

        xp=F.adaptive_avg_pool2d(x,1)
        xp=self.mlp(xp.view(xp.shape[0],-1))
        gamma=self.gamma(xp).unsqueeze(2).unsqueeze(3)
        beta=self.beta(xp).unsqueeze(2).unsqueeze(3)

        for i in range(self.n_res):
            x=getattr(self,f'adapolin_{i}')(x,gamma=gamma,beta=beta)

        x=self.up(x)
        return x,logit,heatmap

    def class_activation_map(self,x,pool_fn,map_fn):
        pooled=pool_fn(x,1)
        logit=map_fn(pooled.view(pooled.shape[0],-1))
        weights=list(map_fn.parameters())[0]
        o=x*weights.unsqueeze(2).unsqueeze(3)
        return o,logit

class Discriminator(nn.Module):
    def __init__(self,base=64,downsample=3):
        super().__init__()
        self.downsample=downsample
        self.base=base
        self.down=[]
        self.down+=[
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(
            in_channels=3,
            out_channels=self.base,
            kernel_size=(4,4),
            stride=(2,2), )),
            nn.LeakyReLU(0.2,inplace=True) ]
        for i in range(self.downsample):
            self.scale=2**(i)
            self.down += [
                nn.ReflectionPad2d(1),
                nn.utils.spectral_norm(nn.Conv2d(
                    in_channels=self.base*self.scale,
                    out_channels=self.base*self.scale*2,
                    kernel_size=(4, 4),
                    stride=(2, 2) if i!=self.downsample-1 else (1,1))),
                nn.LeakyReLU(0.2, inplace=True)]

        self.gmp_fc=nn.utils.spectral_norm(nn.Linear(in_features=self.base*self.scale*2,
                                  out_features=1,bias=False))
        self.gap_fc =nn.utils.spectral_norm(nn.Linear(in_features=self.base * self.scale * 2,
                                out_features=1, bias=False))

        self.conv11=nn.Conv2d(in_channels=self.base*self.scale*4,
                          out_channels=self.base*self.scale*2,
                          kernel_size=(1,1),
                          stride=(1,1))
        self.act=nn.LeakyReLU(0.2,inplace=True)

        self.out_map=nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.utils.spectral_norm(nn.Conv2d(
                in_channels=self.base*self.scale*2,
                out_channels=1,
                kernel_size=(4, 4),
                stride=(1, 1))),
            nn.Tanh()
        )
        self.down=nn.Sequential(*self.down)
    def forward(self,x):
        x=self.down(x)
        gap,gap_logit=self.class_activation_map(x,
                            pool_fn=F.adaptive_avg_pool2d,
                            map_fn=self.gap_fc)
        gmp,gmp_logit=self.class_activation_map(x,
                            pool_fn=F.adaptive_max_pool2d,
                            map_fn=self.gmp_fc)

        x=torch.cat([gap,gmp],dim=1)
        x=self.act(self.conv11(x))
        logit=torch.cat([gap_logit,gmp_logit],dim=1)
        heatmap=torch.sum(x,dim=1)
        x=self.out_map(x)
        return x,logit,heatmap

    def class_activation_map(self,x,pool_fn,map_fn):
        pooled=pool_fn(x,1)
        logit=map_fn(pooled.view(pooled.shape[0],-1))
        weights=list(map_fn.parameters())[0]
        o=x*weights.unsqueeze(2).unsqueeze(3)
        return o,logit

'''
G=Generator()
D_L=Discriminator(base=64)
D_G=Discriminator(downsample=5)
x=torch.randn(1,3,128,128)
x,logit,heatmap=G(x)
dg,logitdg,hdg=D_G(x)
dl,logitdl,hdl=D_L(x)
print(dg.shape)
print(logitdg.shape)
print(hdg.shape)
print(dl.shape)
print(logitdl.shape)
print(hdl.shape)
print(x.shape)
print(logit.shape)
print(heatmap.shape)
'''

