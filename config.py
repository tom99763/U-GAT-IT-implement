batch_size=1
save_iter=5000
lr=1e-4
weight_decay=1e-4
device='cuda'

adv_w=1
cycle_w=10
identity_w=10
cam_w=1000
beta1=0.5
beta2=0.999

image_size=128
image_display=1000
num_gen_img= 20
base=32
n_res=4


load_model=False
train_x_pth='../../data/gan/selfie2anime/train/trainA/'
train_y_pth='../../data/gan/selfie2anime/train/trainB/'
val_x_pth='../../data/gan/selfie2anime/test/testA/'
val_y_pth='../../data/gan/selfie2anime/test/testB/'
save_pth='./save/model.pt'
res_pth='./res'

