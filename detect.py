import cv2
import torch
import torchvision.transforms as transforms
from utils import *
from model import Generator
import config

Gx2y=Generator(base=32,n_res=4).to(config.device)
params=torch.load('./save/model_32base.pt')
Gx2y.load_state_dict(params['Gx2y'])

transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((config.image_size,config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
    ])

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

#cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE); name a window
while(True):
    #capture image
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    h,w,c=frame.shape
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame).unsqueeze(0).to(config.device)
    x2y, _, _ = Gx2y(frame)
    x2y = x2y[0, ...]
    x2y = transform_img(x2y)
    x2y=cv2.resize(x2y,(h,w))
    cv2.imshow('live', x2y)

    #click q to out
    if cv2.waitKey(1) == ord('q'):
        break

# 釋放該攝影機裝置
cap.release()
cv2.destroyAllWindows()