from train_p.resnet import resnet18
from train_p.config import get_config
from PIL.Image import open
import torch as t
from torchvision import transforms
import os
import numpy as np

img_list = ['silent.png','smile.png']
img_list = [os.path.join('./smile_silent',i) for i in img_list]
norm_params = np.loadtxt('record_max_min.txt').T  # 235,2
transform = transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ])
batch = []
for i in img_list:
    img = open(i).convert('RGB')
    img = transform(img)
    batch.append(img)
conf = get_config()
model = resnet18(pretrained=False,**{'num_classes': conf.num_dims})
model.eval()
model.cuda()
for i, img in enumerate(batch):
    out = model(img.unsqueeze(0).cuda())
    out = out.cpu().data.numpy()
    print(out.shape)
    np.savetxt('%d.txt'%(i),out)

