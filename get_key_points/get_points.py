import face_alignment
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as io
import os

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True)

# dataset = r'F:\dataSet\CASIA-maxpy-crop_128'
dataset = r'../dataset/train_aug_120x120'
key_points_dataset = r'../dataset/key_points'
if not os.path.exists(key_points_dataset):
    os.makedirs(key_points_dataset)
dataset = os.path.abspath(dataset)
path = os.listdir(dataset)
path = [os.path.join(dataset,i) for i in path]

#index = path.index(r'HELEN_HELEN_166874696_1_0_8.jpg')
index = 29265

#print(preds)     #字典，key是文件名，值是68,2的float32 points
for i,p in enumerate(path[index:]):
    print(i+index)
    print(p)
    base_name = p.split('/')[-1]
    mat_file = os.path.join(key_points_dataset,base_name)
    preds = fa.get_landmarks(p)
    if preds is None:
        continue
    file_name = os.path.abspath(mat_file).split('.')[0]+'.mat'
    io.savemat(file_name,{'pts':preds[0]})




# fig = plt.figure(figsize=plt.figaspect(.5))
# ax = fig.add_subplot(1, 2, 1)
# ax.imshow(input)
# ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=6,linestyle='-',color='w',lw=2)
# ax.axis('off')
#
# # ax = fig.add_subplot(1, 2, 2, projection='3d')
# # surf = ax.scatter(preds[:,0]*1.2,preds[:,1],preds[:,2],c="cyan", alpha=1.0, edgecolor='b')
# # ax.plot3D(preds[:17,0]*1.2,preds[:17,1], preds[:17,2], color='blue' )
# # ax.plot3D(preds[17:22,0]*1.2,preds[17:22,1],preds[17:22,2], color='blue')
# # ax.plot3D(preds[22:27,0]*1.2,preds[22:27,1],preds[22:27,2], color='blue')
# # ax.plot3D(preds[27:31,0]*1.2,preds[27:31,1],preds[27:31,2], color='blue')
# # ax.plot3D(preds[31:36,0]*1.2,preds[31:36,1],preds[31:36,2], color='blue')
# # ax.plot3D(preds[36:42,0]*1.2,preds[36:42,1],preds[36:42,2], color='blue')
# # ax.plot3D(preds[42:48,0]*1.2,preds[42:48,1],preds[42:48,2], color='blue')
# # ax.plot3D(preds[48:,0]*1.2,preds[48:,1],preds[48:,2], color='blue' )
#
# # ax.view_init(elev=90., azim=90.)
# ax.set_xlim(ax.get_xlim()[::-1])
# plt.show()