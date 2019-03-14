from FaceID_GAN_model import Model
from torchvision.datasets import  ImageFolder
from torch.utils.data import DataLoader
import losses
import torch as t
from torch import optim as opt
from torchvision.transforms import transforms


root_path = r"F:\dataSet\CASIA-maxpy-crop_128"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])
dataset = ImageFolder(root_path,transforms)
model = Model(len(dataset))
optim_c = opt.Adam(filter(lambda x : x.requires_grad is not False, model.c.parameters()),lr=0.0008,weight_decay=0.0005)
optim_d = opt.Adam(filter(lambda x : x.requires_grad is not False, model.d.parameters()),lr=0.0008,weight_decay=0.0005)
optim_g = opt.Adam(filter(lambda x : x.requires_grad is not False, model.g.parameters()),lr=0.0008,weight_decay=0.0005)

model.cuda()
model.train()
k = 0
for step,e in enumerate(range(50)):
    loader = DataLoader(dataset,batch_size=32,shuffle=True,drop_last=True)
    print("%d epoch"%(e+1))
    for data, label in enumerate(loader):
        data = data.cuda()
        r_x_s, r_x_r, f_p_s, f_p_t, f_id_s, f_id_r, c_x_r, c_x_s = model(data)
        lamda = losses.update_lamda(step)
        ld, lc, lg = losses.get_loss(r_x_s, r_x_r, f_p_s,f_p_t, f_id_s, f_id_r, c_x_r, c_x_s,label,k,lamda)
        k = losses.update_k(k, r_x_r, r_x_s)

        optim_d.zero_grad()
        ld.backward(retain_grapg=True)
        optim_d.step()

        optim_c.zero_grad()
        lc.backward(retain_graph=True)
        optim_c.step()

        optim_g.zero_grad()
        lg.backward(retain_grad=True)
        optim_g.step()

        if (step+1)% 50000 ==0:
            for param_group in optim_g.param_groups:
                param_group['lr'] = param_group['lr'] - 0.0002
            for param_group in optim_c.param_groups:
                param_group['lr'] = param_group['lr'] - 0.0002
            for param_group in optim_d.param_groups:
                param_group['lr'] = param_group['lr'] - 0.0002

    if optim_g.param_groups[0]['lr'] <= 0:
        break
