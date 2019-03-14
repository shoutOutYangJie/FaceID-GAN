# from train_p.config import get_config
# import scipy.io as io
# import numpy as np
# import os
# import random
# conf = get_config()
# label_list = os.listdir(conf.vector_label)
# label_path = [os.path.join(conf.vector_label,i) for i in label_list]
# label_path = random.sample(label_path,10001)
# total_mat = np.zeros([235,1])
# interval = 1000
# start = 0
#
# max = np.full((235,),-9999999)
# min = np.full((235,),9999999)
# while(start<len(label_path)):
#
#     for path in label_path[start:start+interval]:
#         mat = io.loadmat(path)['saved_vector']
#         mat = np.array(mat)
#         total_mat = np.hstack((total_mat,mat))
#
#     max_temp = total_mat.max(axis = 1)
#     min_temp = total_mat.min(axis =1)
#     max = np.maximum(max,max_temp)
#     min = np.minimum(min,min_temp)
#     start = start + interval
#     print(start)
#
#
#
# content = np.vstack((max,min))
# np.savetxt('record_max_min.txt',content)

# a = np.loadtxt('record_max_min.txt')
# print(a.shape)

# path = r'F:\dataSet\Face_vector'
# label_list = os.listdir(path)
# f = open('lable_list.txt','w')
# for label in label_list:
#     f.write(os.path.join(path,label)+'\r')
#
# f = open('label_list.txt','r')
# a = f.readlines()
# for i in a:
#     print(i.strip())

# vector_label = r'F:\dataSet\Face_vector'
# v_list = os.listdir(vector_label)[0]
# a = io.loadmat(os.path.join(vector_label,v_list))['saved_vector']
# a = np.array(a)
# print(a.shape)

from train_p.resnet import resnet18
from train_p.config import get_config
from train_p.vector_loader import get_batch
from torch import nn,optim
import torch as t
import os

conf = get_config()
train_loader = get_batch(conf)
# val_loader = get_batch(conf, is_training= False)

model = resnet18(pretrained=False,**{'num_classes': conf.num_dims})

criterion = nn.MSELoss()
if conf.resume:
    ckpt = t.load(conf.saved_model)
    model.load_state_dict(ckpt['state_dict'])
    start = 0
else:
    start = 1


model.cuda()
for epoch in range(start,conf.epochs+1):
    model.eval()
    print('the %dth epoch on training'%(epoch))
    for i, batch in enumerate(train_loader):
        model.zero_grad()
        img,vector = batch[0].cuda(), batch[1].cuda()
        out = model(img)   #batch_size,235
        # calculate loss using WPPC loss


        loss = criterion(out,vector)

        print(loss.item())




