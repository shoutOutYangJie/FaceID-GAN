from train_p.resnet import resnet18
from train_p.config import get_config
from train_p.vector_loader import get_batch
from torch import nn,optim
import torch as t
import os
'''
  saved_vector = [Pose_Para;Shape_Para;Exp_Para];
  235-dims vector is as above formulation.
  Pose_Para: [phi, gamma, theta, t1,t2,t3, f]
  Shape_Para: 199-dims
  Exp_Para: 29-dims
'''
conf = get_config()
train_loader = get_batch(conf)
# val_loader = get_batch(conf, is_training= False)

model = resnet18(pretrained=False,**{'num_classes': conf.num_dims})

optim = optim.Adam(model.parameters(),lr=conf.lr,weight_decay=0.0005)
criterion = nn.MSELoss()
if conf.resume:
    ckpt = t.load(conf.saved_model)
    model.load_state_dict(ckpt['state_dict'])
    start = ckpt['epoch']+1
else:
    start = 1


model.cuda()
for epoch in range(start,conf.epochs+1):
    model.train()
    print('the %dth epoch on training'%(epoch))
    for i, batch in enumerate(train_loader):
        model.zero_grad()
        img,vector = batch[0].cuda(), batch[1].cuda()
        out = model(img)   #batch_size,235
        # calculate loss using WPPC loss


        loss = criterion(out,vector)
        print('%dth iteration loss: '%(i),loss.item())
        loss.backward()
        optim.step()

    # if epoch %conf.freq_validation:
    #     model.eval()
    #     with t.no_grad():
    #         for data, vector in val_loader:
    #             data, vector = data.cuda(), vector.cuda()
    #             out = model(data)
    #             loss = criterion(out,vector)
    #             print('val loss is %f'%(loss.item()))

    # if epoch%conf.freq_saved_model:
    state_dict = model.state_dict()

    t.save({
        'epoch':epoch,
        'state_dict':state_dict,
    },os.path.join(conf.saved_model, '%d.pth' % epoch))





