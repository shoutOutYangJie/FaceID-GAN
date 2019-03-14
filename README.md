# FaceID-GAN
this is a re-implementation of CVPR2018 paper "FaceID-GAN" using Pytorch

note: I don't finish this work because model is so big that my compute doesn't support training, but I haved done most of work such as training a P net according to paper and build model arctiture. Maybe you need optimizer my code, because I don't test it.

# dependence
* Pytorch >=0.4
* python3
* pip install face-alignment

# get pre-trained P net
* firstly to get 68 key points of one face, using this [great work](https://github.com/1adrianb/face-alignment)
* using [open source code](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/HPEN/main.htm) to get 3DMM parameters for training P net

# acknowledgement
* thanks for first author of FaceID-GAN to reply my email and untie my confusion
* thanks for the wonderful open source works to develop this project
