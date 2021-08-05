import torch.nn as nn
import torch
import cv2
import numpy as np
import config

mse=nn.MSELoss().to(config.device)
mae=nn.L1Loss().to(config.device)
bce=nn.BCEWithLogitsLoss().to(config.device)


def d_loss(real,fake):
    return mse(real,torch.ones_like(real))+mse(fake,torch.zeros_like(fake))

def g_loss(fake):
    return mse(fake,torch.ones_like(fake))

def recon_loss(x,x_recon):
    return mae(x_recon,x)

def cam_loss(a2b,b2b):
    return bce(a2b,torch.ones_like(a2b))+bce(b2b,torch.zeros_like(b2b))

def transform_img(x):
    x=x.detach().cpu().numpy().transpose(1,2,0)
    x=x*127.5+127.5
    x=x.astype('uint8')
    x=cv2.cvtColor(x,cv2.COLOR_RGB2BGR)
    return x

def transform_heatmap(x, size):
    x = x.detach().cpu().numpy()[...,np.newaxis]
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img


def gen_img(pth,image_size,x_loader,y_loader,n,Gx2y,Gy2x):
    X2Y=np.zeros((image_size*7,0,3))
    Y2X=np.zeros((image_size*7,0,3))
    for idx,(x,y) in enumerate(zip(x_loader,y_loader)):
        if idx==n:
            break
        x, y = x[0].to(config.device), y[0].to(config.device)
        x2y,_,x2y_heatmap=Gx2y(x)
        y2x,_,y2x_heatmap=Gy2x(y)
        x2x,_,x2x_heatmap=Gy2x(x)
        y2y,_,y2y_heatmap=Gx2y(y)
        x2y2x,_,x2y2x_heatmap=Gy2x(x2y)
        y2x2y,_,y2x2y_heatmap=Gx2y(y2x)


        X2Y=np.concatenate([X2Y,np.concatenate([
            transform_img(x[0]),transform_heatmap(x2y_heatmap[0],image_size),
            transform_img(x2y[0]),transform_heatmap(x2x_heatmap[0],image_size),
            transform_img(x2x[0]),transform_heatmap(x2y2x_heatmap[0],image_size),
            transform_img(x2y2x[0])],0)],1)


        Y2X = np.concatenate([Y2X, np.concatenate([
            transform_img(y[0]), transform_heatmap(y2x_heatmap[0], image_size),
            transform_img(y2x[0]), transform_heatmap(y2y_heatmap[0], image_size),
            transform_img(y2y[0]), transform_heatmap(y2x2y_heatmap[0], image_size),
            transform_img(y2x2y[0])], 0)], 1)

    cv2.imwrite(f'{pth}/X2Y.jpg',X2Y)
    cv2.imwrite(f'{pth}/Y2X.jpg',Y2X)


def save(pth,Gx2y,Gy2x,Dgx,Dgy,Dlx,Dly):
    params={}
    params['Gx2y']=Gx2y.state_dict()
    params['Gy2x'] = Gy2x.state_dict()
    params['Dgx'] = Dgx.state_dict()
    params['Dgy'] = Dgy.state_dict()
    params['Dlx'] = Dlx.state_dict()
    params['Dly'] = Dly.state_dict()
    torch.save(params,pth)

def load(pth,Gx2y,Gy2x,Dgx,Dgy,Dlx,Dly):
    params=torch.load(pth)
    Gx2y.load_state_dict(params['Gx2y'])
    Gy2x.load_state_dict(params['Gy2x'])
    Dgx.load_state_dict(params['Dgx'])
    Dgy.load_state_dict(params['Dgy'])
    Dlx.load_state_dict(params['Dlx'])
    Dly.load_state_dict(params['Dly'])


def rho_clipper(m):
    if hasattr(m, 'rho'):
        rho = m.rho.data
        rho = torch.clamp(rho, 0., 1.)
        m.rho.data = rho