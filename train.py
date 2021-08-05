import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import config
from model import Generator,Discriminator
import itertools
from utils import *
from tqdm import tqdm


def flow(x,y, Gx2y, Gy2x, Dgx, Dgy, Dlx, Dly,G_opt,D_opt):
    D_opt.zero_grad()
    #Discriminator update
    x2y, _, x2y_heatmap = Gx2y(x)
    y2x, _, y2x_heatmap = Gy2x(y)

    y_dgy,y_dgy_logit,_=Dgy(y)
    x2y_dgy, x2y_dgy_logit, _ = Dgy(x2y)
    x_dgx,x_dgx_logit,_=Dgx(x)
    y2x_dgx,y2x_dgx_logit,_=Dgx(y2x)

    y_dly,y_dly_logit,_=Dly(y)
    x2y_dly, x2y_dly_logit, _ = Dly(x2y)
    x_dlx,x_dlx_logit,_=Dlx(x)
    y2x_dlx,y2x_dlx_logit,_=Dlx(y2x)

    dx_adv_loss=config.adv_w*(d_loss(x_dgx,y2x_dgx)+d_loss(x_dgx_logit,y2x_dgx_logit)+
                              d_loss(x_dlx,y2x_dlx)+d_loss(x_dlx_logit,y2x_dlx_logit))

    dy_adv_loss=config.adv_w*(d_loss(y_dgy,x2y_dgy)+d_loss(y_dgy_logit,x2y_dgy_logit)+
                              d_loss(y_dly,x2y_dly)+d_loss(y_dly_logit,x2y_dly_logit))

    d_adv_loss=dx_adv_loss+dy_adv_loss
    d_adv_loss.backward()
    D_opt.step()

    G_opt.zero_grad()
    x2y, x2y_logit, x2y_heatmap = Gx2y(x)
    y2x, y2x_logit, y2x_heatmap = Gy2x(y)
    x2x, x2x_logit, x2x_heatmap = Gy2x(x)
    y2y, y2y_logit, y2y_heatmap = Gx2y(y)
    x2y2x, _, x2y2x_heatmap = Gy2x(x2y)
    y2x2y, _, y2x2y_heatmap = Gx2y(y2x)
    y_dgy,y_dgy_logit,_=Dgy(y)
    x2y_dgy, x2y_dgy_logit, _ = Dgy(x2y)
    x_dgx,x_dgx_logit,_=Dgx(x)
    y2x_dgx,y2x_dgx_logit,_=Dgx(y2x)

    y_dly,y_dly_logit,_=Dly(y)
    x2y_dly, x2y_dly_logit, _ = Dly(x2y)
    x_dlx,x_dlx_logit,_=Dlx(x)
    y2x_dlx,y2x_dlx_logit,_=Dlx(y2x)

    gx2y_loss=config.adv_w*(g_loss(x2y_dgy)+g_loss(x2y_dly)+g_loss(x2y_dgy_logit)+g_loss(x2y_dly_logit))+\
        config.identity_w*recon_loss(y,y2y)+config.cycle_w*recon_loss(y,y2x2y)+\
        config.cam_w*cam_loss(x2y_logit,y2y_logit)
    gy2x_loss = config.adv_w * (g_loss(y2x_dgx) + g_loss(y2x_dlx) + g_loss(y2x_dgx_logit) + g_loss(y2x_dlx_logit))+\
        config.identity_w*recon_loss(x,x2x)+config.cycle_w*recon_loss(x,x2y2x)+\
        config.cam_w*cam_loss(y2x_logit,x2x_logit)

    g_total_loss=gx2y_loss+gy2x_loss
    g_total_loss.backward()
    G_opt.step()

    Gx2y.apply(rho_clipper)
    Gy2x.apply(rho_clipper)

    return g_total_loss.detach().cpu().numpy(),d_adv_loss.detach().cpu().numpy()



def main():
    train_transfrom=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(config.image_size+30,config.image_size+30)),
        transforms.RandomCrop(size=config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

    test_transfrom = transforms.Compose([
        transforms.Resize(size=(config.image_size , config.image_size )),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

    trainx=ImageFolder(root=config.train_x_pth,transform=train_transfrom)
    trainy = ImageFolder(root=config.train_y_pth, transform=train_transfrom)
    testx = ImageFolder(root=config.val_x_pth, transform=test_transfrom)
    testy = ImageFolder(root=config.val_y_pth, transform=test_transfrom)

    train_x_loader=DataLoader(dataset=trainx,batch_size=config.batch_size,shuffle=True)
    train_y_loader = DataLoader(dataset=trainy, batch_size=config.batch_size, shuffle=True)
    test_x_loader = DataLoader(dataset=testx, batch_size=config.batch_size, shuffle=False)
    test_y_loader = DataLoader(dataset=testy, batch_size=config.batch_size, shuffle=False)

    Gx2y=Generator(base=config.base,n_res=config.n_res).to(config.device)
    Gy2x =Generator(base=config.base, n_res=config.n_res).to(config.device)
    Dlx=Discriminator(base=config.base).to(config.device)
    Dly=Discriminator(base=config.base).to(config.device)
    Dgx=Discriminator(base=config.base,downsample=5).to(config.device)
    Dgy=Discriminator(base=config.base,downsample=5).to(config.device)

    G_opt=optim.Adam(params=itertools.chain(Gx2y.parameters(),Gy2x.parameters()),
                     lr=config.lr,betas=(config.beta1,config.beta2),weight_decay=config.weight_decay)
    
    D_opt=optim.Adam(params=itertools.chain(Dgx.parameters(),Dgy.parameters(),Dlx.parameters(),Dly.parameters()),
                     lr=config.lr,betas=(config.beta1,config.beta2),weight_decay=config.weight_decay)


    if config.load_model:
        load(config.save_pth,Gx2y,Gy2x,Dgx,Dgy,Dlx,Dly)
        print('---load state dic---')

    Gx2y.train(),Gy2x.train(),Dgx.train(),Dgy.train(),Dlx.train(),Dly.train()
    i=0
    while True:
        loop=tqdm(zip(train_x_loader, train_y_loader),leave=True)
        for x, y in loop:
            x, y = x[0].to(config.device), y[0].to(config.device)
            G_loss,D_loss=flow(x,y, Gx2y, Gy2x, Dgx, Dgy, Dlx, Dly,G_opt,D_opt)
            loop.set_postfix(loss=f'G_loss:{G_loss} D_loss:{D_loss}')
            if i%config.image_display==0:
                Gx2y.eval(), Gy2x.eval()
                gen_img(config.res_pth,config.image_size,test_x_loader, test_y_loader, config.num_gen_img, Gx2y, Gy2x)
                Gx2y.train(), Gy2x.train()
            i+=1
            if i%config.save_iter==0:
                save(config.save_pth, Gx2y, Gy2x, Dgx, Dgy, Dlx, Dly)

if __name__ == '__main__':
    main()