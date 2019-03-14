from BeGAN.began import G,D
from Classifier.resnet import resnet50 as C
from train_p.resnet import resnet18 as P
import torch as t
from torch import nn
import numpy as np
def initial_model(class_num):
    classifer = C(pretrained=False,**{'num_classes':class_num})
    p = P(pretrained=False,**{'num_classes':235})
    g = G(h=613,n=64,output_dim=(3,128,128))
    d = D(h=613,n=64,input_dim=(3,128,128))
    total_num_params = 0
    for m in [classifer,p,g,d]:
        for p in m.parameters():
            total_num_params += p.numel()
    print('number of all models\' parameters are %d'%(total_num_params))
    return classifer,p,g,d

def load_pth(p,p_path='./train_p/saved_model/2.pth'):
    p.load_state_dict(t.load(p_path))
    return p

def transform_func(v,smile,silent):
     weight = np.random.uniform(0,1)
     yaw_angle = t.Tensor(v.shape[0],1).uniform_(-0.3,0.3)
     new_exp_v = t.lerp(smile,silent,weight)
     v[:,-29:] = new_exp_v
     v[:,1] = yaw_angle
     # v = t.cat((yaw_angle,v[:,-228:]))
     return v

class Model(nn.Module):
    def __init__(self,people_num):
        super(Model,self).__init__()
        self.c, self.p, self.g, self.d = initial_model(people_num)
        self.p = load_pth(self.p)
        for i in self.p.parameters():
            i.requires_grad = False
        self.smile_vector = t.from_numpy(np.loadtxt('./train_p/1.txt',dtype=np.float32))[-29:]
        self.silent_vector = t.from_numpy(np.loadtxt('./train_p/0.txt',dtype=np.float32))[-29:]
        temp_vector = [0]+[1]+[0]*5 + [1]*228
        self.temp_vector = t.from_numpy(np.array(temp_vector, dtype=np.float32))
    def forward(self, x):
        b,c,h,w = x.shape
        c_x_r,f_id_r = self.c(x)
        f_p_r = self.p(x)
        f_p_r = f_p_r.mul(self.temp_vector)
        z = t.Tensor(b,128).uniform_(-1,1)
        f_p_t = transform_func(f_p_r,self.smile_vector,self.silent_vector)   # 229-dims
        g_inputs = t.cat([f_id_r,f_p_t,z],axis=1)
        xs = self.g(g_inputs)
        r_x_s = t.dist(self.d(xs),xs,p=1)   #还是不要加detach()了，三部分不像两部分。而且C在前面也有
        f_p_s = self.p(xs)
        f_p_s = f_p_s.mul(self.temp_vector)
        c_x_s, f_id_s = self.c(xs)
        r_x_r = t.dist(self.d(x),x,p=1)
        return r_x_s, r_x_r, f_p_s,f_p_t, f_id_s, f_id_r, c_x_r, c_x_s



if __name__ =='__main__':
    initial_model()
    # a = np.loadtxt('./train_p/0.txt')
    # print(a.shape)   # 235,

