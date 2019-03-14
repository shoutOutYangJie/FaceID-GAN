import torch as t
from torch.nn import CrossEntropyLoss

CE = CrossEntropyLoss()
lamda_1 = 0.4
lamda_2 = 0.3
lamda_3 = 0.3

def get_loss(r_x_s, r_x_r, f_p_s,f_p_t, f_id_s, f_id_r, c_x_r, c_x_s,label,k,lamda):
    ld = r_x_r-k*r_x_s
    lc = CE(c_x_r,label) + lamda*CE(c_x_s,label)
    lg = lamda_1*r_x_s + lamda_2*cosine_distance(f_id_r,f_id_s.t()) +lamda_3*l2_distance(f_p_s,f_p_t)
    return ld,lc,lg

def cosine_distance(f1,f2):
    f1_norm = t.norm(f1,p=2,dim=1)
    f2_norm = t.norm(f2,p=2,dim=1)
    norm = f1_norm.mul(f2_norm)
    print('norm shape ',norm.shape)
    d = 1 - t.mm(f1,f2)/norm.unsqueeze(1)
    return t.sum(t.diag(d,0))

def l2_distance(f1,f2):
    return t.sum(t.pow((f1-f2),2))


def update_k(k,r_x_r,r_x_s):
    return k + 0.001*(0.4*r_x_r.cpu().data.numpy()-r_x_s.cpu().data.numpy())

def update_lamda(iters):
    if iters <= 30000:
        return 0.9
    elif iters <= 60000:
        return 0.7
    elif iters <= 90000:
        return 0.5
    elif iters <= 120000:
        return 0.3
    elif iters <= 150000:
        return 0.15
    else:
        return 0.05

