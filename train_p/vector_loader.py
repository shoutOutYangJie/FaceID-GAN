import os
import numpy as np
from scipy import io
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch as t
from PIL import Image

class P_data_loader(Dataset):
    def __init__(self,conf,transform,is_training = True):
        super(P_data_loader,self).__init__()
        self.conf = conf
        self.root = conf.vector_3dmm_list_for_train
        with open(self.root,'r') as f:
            data = [ line.strip() for line in f.readlines()]    # 记录data的绝对地址
        self.label = np.random.permutation(data)[:10000]
        self.transform = transform
        self.norm_params = np.loadtxt('record_max_min.txt').T   # 235,2

    def __len__(self):
        return len(self.label)

    def __getitem__(self, inx):
        path = self.label[inx]   #.mat
        base_name = os.path.basename(path).split('.')[0]
        base_name = base_name + '.jpg'
        # print(base_name)
        image_file = os.path.join(self.conf.image_dataset,base_name)
        # print(self.conf.image_dataset)
        img = Image.open(image_file).convert('RGB')
        img = self.transform(img)
        vector = io.loadmat(path)['saved_vector'] # id exp pose   dims = 235
        vector = np.array(vector)    # shape 是多少,待测试     235,1
        vector = self.normalization(vector).squeeze().astype(np.float32)
        # img = t.from_numpy(img)   #transform did this
        vector = t.from_numpy(vector)   # (235,)
        # print(vector.shape)
        return [img, vector]

    def normalization(self,vector):
        temp = (vector-self.norm_params[:,1].reshape(-1,1))/(self.norm_params[:,0]-self.norm_params[:,1]).reshape(-1,1)
        return temp


def get_batch(conf,):

    dataset = P_data_loader(conf,transform=transforms.Compose([
                                     # transforms.Resize((110,110)),
                                     # transforms.RandomCrop((96,96)),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                            ]),
                           )
    dataloader = DataLoader(dataset, batch_size=conf.batch_size,
                            shuffle=False, drop_last=True)  # drop_last is necessary,because last iteration may fail
    return dataloader

